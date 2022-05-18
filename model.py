import torch
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
r"""
    На двумерной плоскости раскиданы n точек;
    Триангулируем эти точки, создавая матрицу смежности для графового слоя;
    
    Используем loss = (ddf/dxdx - ddf/dydy) ^ 2 + (df(x, 0))^2 + (f(x, 0))^2
    
    Ну и типа решили уравнение 
"""


class GNN(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.X = data[:, :-1]
        self.U = torch.Tensor(np.diag(data[:, -1].flatten())).cuda()
        self.lin1 = torch.nn.Linear(2, data.shape[0])
        self.lin2 = torch.nn.Linear(data.shape[0], 1)
        self.A = None

    def generate_adjacency(self):
        tri = Delaunay(self.X)
        g = nx.Graph()
        for path in tri.simplices:
            nx.add_path(g, path)
        self.A = torch.Tensor(nx.to_numpy_array(g)).cuda()
        print("-----------------------------------------")
        print("Adjacency matrix generated succesfully!!!")
        print("-----------------------------------------")

    def forward(self, x, y):
        x = torch.cat([x, y])
        h1 = self.lin1(x)           # W_1x
        h2 = self.A @ self.U @ h1   # AUW_1x
        h3 = torch.sigmoid(h2)      # sigmoid(AUW_1x)
        h4 = self.lin2(h3)          # W_2sigmoid(AUW_1x)
        h5 = torch.sigmoid(h4)
        return h5


def train(num_epochs: int,
          model,
          loss_fn,
          opt,
          train_dl: torch.utils.data.DataLoader):
    losses = []
    losses_per_epochs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.generate_adjacency()
    for epoch in range(num_epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device).requires_grad_(), yb.to(device).requires_grad_()
            opt.zero_grad()
            loss = loss_fn(xb, yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        losses_per_epochs.append(sum(losses[-i:])/i)
        print(f'Epoch: {epoch} | Loss: {sum(losses[-i:])/i}')
    return losses


class phys_loss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y):
        r"""
        loss = (ddf/dxdx - ddf/dydy) ^ 2 + (f(x, 0))^2 + (df/dt(x, 0))^2 -> 0
        :param x:
        :return: loss
        """

        u = self.model(x, y)
        dx = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        dy = torch.autograd.grad(u, y, create_graph=True, retain_graph=True, allow_unused=True)[0]
        ddx = torch.autograd.grad(dx, x,  retain_graph=True, allow_unused=True)[0]
        ddy = torch.autograd.grad(dy, y,  retain_graph=True, allow_unused=True)[0]
        u1 = self.model(torch.Tensor([0]).cuda(), y)
        loss = (ddx - ddy) ** 2 + u1 ** 2
        y1 = torch.zeros(1, requires_grad=True).cuda()
        u2 = self.model(x, y1)

        dy1 = torch.autograd.grad(u2, y1, create_graph=True, retain_graph=True, allow_unused=True)[0]
        loss = loss + (dy1 - torch.sin(x)) ** 2
        return loss


def get_dataloader(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    tds = TensorDataset(x, y)
    tdl = DataLoader(tds)
    return tdl
