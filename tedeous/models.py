import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Fourier_embedding(nn.Module):
    """
    Class for Fourier features generation.

    Args:
        L: list[float or None], sin(w*x)/cos(w*X) frequencie parameter, w = 2*pi/L
        M: list[float or None], number of (sin, cos) pairs in result embedding
        ones: bool, enter or not ones vector in result embedding.

    Examples:
        u(t,x) if user wants to create 5 Fourier features in 'x' direction with L=5:
            L=[None, 5], M=[None, 5].
    """

    def __init__(self, L=[1], M=[1], ones=False):
        super().__init__()
        self.M = M
        self.L = L
        self.idx = [i for i in range(len(self.M)) if self.M[i] is None]
        self.ones = ones
        self.in_features = len(M)
        not_none = sum([i for i in M if i is not None])
        is_none = self.M.count(None)
        if is_none == 0:
            self.out_features = not_none * 2 + self.in_features
        else:
            self.out_features = not_none * 2 + is_none
        if ones is not False:
            self.out_features += 1

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward method for Fourier features generation.

        Args:
            grid: calculation domain.
        Returns:
            out: embedding with Fourier features.
        """
        if self.idx == []:
            out = grid
        else:
            out = grid[:, self.idx]

        for i in range(len(self.M)):
            if self.M[i] is not None:
                Mi = self.M[i]
                Li = self.L[i]
                w = 2.0 * np.pi / Li
                k = torch.arange(1, Mi + 1).reshape(-1, 1).float()
                x = grid[:, i].reshape(1, -1)
                x = (k @ x).T
                embed_cos = torch.cos(w * x)
                embed_sin = torch.sin(w * x)
                out = torch.hstack((out, embed_cos, embed_sin))

        if self.ones is not False:
            out = torch.hstack((out, torch.ones_like(out[:, 0:1])))

        return out


class FourierNN(nn.Module):
    """
    Class for realizing neural network with Fourier features
    and skip connection.

    Args:
        L: list[float or None], sin(w*x)/cos(w*X) frequency parameter, w = 2*pi/L.
        M: list[float or None], number of (sin, cos) pairs in result embedding.
        activation: nn.Module object, activation function.
        ones: bool, enter or not ones vector in result embedding.
    """

    def __init__(self, layers=[100, 100, 100, 1], L=[1], M=[1],
                 activation=nn.Tanh(), ones=False):
        """
            Class for realizing neural network with Fourier features
            and skip connection.

            Args:
                L: list[float or None], sin(w*x)/cos(w*X) frequency parameter, w = 2*pi/L.
                M: list[float or None], number of (sin, cos) pairs in result embedding.
                activation: nn.Module object, activation function.
                ones: bool, enter or not ones vector in result embedding.
            """
        super(FourierNN, self).__init__()
        self.L = L
        self.M = M
        FFL = Fourier_embedding(L=L, M=M, ones=ones)

        layers = [FFL.out_features] + layers

        self.linear_u = nn.Linear(layers[0], layers[1])
        self.linear_v = nn.Linear(layers[0], layers[1])

        self.activation = activation
        self.model = nn.ModuleList([FFL])
        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for neural network.

        Args:
            grid: calculation domain.
        Returns:
            predicted values.

        """
        input = self.model[0](grid)
        V = self.activation(self.linear_v(input))
        U = self.activation(self.linear_u(input))
        for layer in self.model[1:-1]:
            output = self.activation(layer(input))
            input = output * U + (1 - output) * V

        output = self.model[-1](input)

        return output


class FeedForward(nn.Module):
    def __init__(self, layers, activation, parameters=None):
        super(FeedForward, self).__init__()
        model = []
        for i in range(len(layers)-2):
            model.append(nn.Linear(layers[i], layers[i+1]))
            model.append(activation)
        model.append(nn.Linear(layers[-2], layers[-1]))
        self.net = torch.nn.Sequential(*model)
        if parameters is not None:
            self.reg_param(parameters)

    def forward(self, x):
        return self.net(x)

    def reg_param(self, parameters: dict):
        for key, value in parameters.items():
            parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                           requires_grad=True).float())
            self.net.register_parameter(key, parameters[key])


def parameter_registr(model, parameters):
    for key, value in parameters.items():
        parameters[key] = torch.nn.Parameter(torch.tensor([value],
                                        requires_grad=True).float())
        model.register_parameter(key, parameters[key])


def mat_model(grid, equation, nn_model=None):
    if type(equation) is list:
        eq_num = len(equation)
    else:
        eq_num = 1

    shape = [eq_num] + list(grid.shape)[1:]

    if nn_model != None:
        nn_grid = torch.vstack([grid[i].reshape(-1) for i in \
                                range(grid.shape[0])]).T.float()
        model = nn_model(nn_grid).detach()
        model = model.reshape(shape)
    else:
        model = torch.ones(shape)

    return model