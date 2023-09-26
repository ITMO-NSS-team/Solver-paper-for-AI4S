import deepxde as dde
from deepxde.backend import tf
import tensorflow as tf

import numpy as np
from scipy.io import loadmat
from scipy.integrate import quad
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import time
import pandas as pd
import seaborn as sns

import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')
sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solution import Solution
from tedeous.device import solver_device, device_type
from tedeous.models import FourierNN


def sln_ac(grid):
    data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'reference_sln/AC.mat')))
    usol = data['uu'].reshape(-1)
    t_star = data['tt'][0]
    x_star = data['x'][0]
    grid_data = torch.cartesian_prod(torch.from_numpy(x_star),
                                     torch.from_numpy(t_star)).float()
    u = scipy.interpolate.griddata(grid_data, usol, grid.to('cpu'), method='nearest')

    return torch.from_numpy(u.reshape(-1))


def solution_plot(model):

    plt.rc('font', size= 15 )

    solver_device('cpu')
    model = model.to('cpu')

    x = torch.from_numpy(np.linspace(-1, 1, 61, dtype=np.float64))

    fig = plt.figure(figsize=(18,9))

    for i, time in enumerate([0, 0.5, 1.]):
        t = torch.from_numpy(np.array([time], dtype=np.float64))
        grid_pred = torch.cartesian_prod(t, x).float()
        grid_exact = torch.cartesian_prod(x, t).float()
        u_exact = sln_ac(grid_exact)
        u_pred = model(grid_pred)

        ax1 = fig.add_subplot(1, 3, i+1)
        if time ==1.:
            ax1.plot(x, u_exact, linewidth=4, label='exact')
            ax1.plot(x, u_pred.detach().cpu().numpy().reshape(-1), '--', linewidth=4, label='numerical')
        else:
            ax1.plot(x, u_exact, linewidth=4)
            ax1.plot(x, u_pred.detach().cpu().numpy().reshape(-1), '--', linewidth=4)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.set_title('t={}'.format(time))
        ax1.grid()

    ax1.legend(bbox_to_anchor=(-0.65, -0.15), loc='lower center', mode=" expand", ncol=2)
    plt.savefig('AC.pdf', dpi=1000, bbox_inches='tight')


def solver_ac(grid_res, cache, iterations, save_sln):
    exp_dict_list = []
    solver_device('gpu')
    x = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))

    grid = torch.cartesian_prod(t, x).float()

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), x).float()
        
    bndval1 = bnd1[:,1:]**2*torch.cos(np.pi*bnd1[:,1:])
        
    bnd2_l = torch.cartesian_prod(t, torch.from_numpy(np.array([-1], dtype=np.float64))).float()

    bnd2_r = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()

    bnd2 = [bnd2_l, bnd2_r]

    bnd3_l = torch.cartesian_prod(t, torch.from_numpy(np.array([-1], dtype=np.float64))).float()

    bnd3_r = torch.cartesian_prod(t, torch.from_numpy(np.array([1], dtype=np.float64))).float()

    bop3= {
            'du/dx':
                {
                    'coeff': 1,
                    'du/dx': [1],
                    'pow': 1,
                    'var': 0
                }
    }

    bnd3 = [bnd3_l, bnd3_r]
        
    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'],
            [bnd2, 'periodic'],
            [bnd3, bop3, 'periodic']]

    AC = {
        '1*du/dt**1':
            {
                'coeff': 1,
                'du/dt': [0],
                'pow': 1,
                'var': 0
            },
        '-0.0001*d2u/dx2**1':
            {
                'coeff': -0.0001,
                'd2u/dx2': [1,1],
                'pow': 1,
                'var': 0
            },
        '+5u**3':
            {
                'coeff': 5,
                'u': [None],
                'pow': 3,
                'var': 0
            },
        '-5u**1':
            {
                'coeff': -5,
                'u': [None],
                'pow': 1,
                'var': 0
            }
    }
        
    model = FourierNN([128, 128, 128, 1], L=[None, 2], M=[None, 10])


    start = time.time()

    equation = Equation(grid, AC, bconds).set_strategy('autograd')

    model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100, verbose=1, learning_rate=1e-3,
                                                            eps=1e-6, tmax=iterations, use_cache=cache,
                                                            patience=3, tol=100, save_always=cache, print_every=None)

    end = time.time()
    time_part = end - start

    device = device_type()

    x1 = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t1 = torch.from_numpy(np.linspace(0, 1, grid_res + 1))
    grid1 = torch.cartesian_prod(x1, t1).float().to(device)

    u_exact = sln_ac(grid1).to(device)

    predict = torch.t(model(grid.to(device)).reshape(grid_res+1, grid_res+1)).reshape(-1)

    error_rmse = torch.sqrt(torch.mean((u_exact - predict) ** 2))
    
    exp_dict_list.append({'grid_res': grid_res, 'time': time_part, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'solver': 'tedeous', 'cache': cache})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    if save_sln:
        solution_plot(model)

    return exp_dict_list


def deepxde_ac(grid_res, optimizer, iterations):
    exp_dict_list = []
    solver_device('cpu')
    start = time.time()
    domain = (grid_res + 1) ** 2 - (grid_res + 1) * 4
    bound = (grid_res + 1) * 4
    init = (grid_res + 1) * 1

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    d = 0.0001

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - d * dy_xx - 5 * (y - y**3)

    def output_transform(x, y):
        return x[:, 0:1]**2 * tf.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=domain, num_boundary=bound, num_initial=init)
    net = dde.nn.FNN([2] + [128] * 3 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    if type(optimizer) is list:
        model.compile('adam', lr=1e-3)
        model.train(iterations=iterations)
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
    else:
        model.compile('adam', lr=1e-3)
        model.train(iterations=iterations)

    end = time.time()
    time_part = end - start

    x = torch.from_numpy(np.linspace(-1, 1, grid_res + 1))
    t = torch.from_numpy(np.linspace(0, 1, grid_res + 1))

    grid = torch.cartesian_prod(x, t).float()

    u_exact = sln_ac(grid)
    error_rmse = torch.sqrt(torch.mean((u_exact - model.predict(grid).reshape(-1)) ** 2))

    exp_dict_list.append(
        {'grid_res': grid_res, 'time': time_part, 'RMSE': error_rmse.detach().numpy(), 'solver': 'deepxde',
         'optimizer': len(optimizer)})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))
    return exp_dict_list


nruns = 10
###########################
exp_dict_list = []

cache = False

for grid_res in range(30, 71, 10):
    for i in range(nruns):
        if grid_res==60 and i==0:
            save_sln = True
        else:
            save_sln = False
        exp_dict_list.append(solver_ac(grid_res, cache, 15000, save_sln))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('solver_AC.csv')
############################

###########################
exp_dict_list = []

for grid_res in range(30, 71, 10):
    for _ in range(nruns):
        exp_dict_list.append(deepxde_ac(grid_res, 'Adam', 15000))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('deepxde_AC.csv')
###########################

df1 = pd.read_csv ('deepxde_AC.csv')
df2 = pd.read_csv ('solver_AC.csv')
df = pd.concat((df1,df2))

plt.figure(figsize=(18,8))
plt.rc('font', size= 21 )
plt.subplot(1,2,1)
plt.grid(True)
mx1 = df1.groupby('grid_res')['RMSE'].median()
mx2 = df2.groupby('grid_res')['RMSE'].median()
sns.boxplot(x='grid_res',y='RMSE', data=df, hue='solver', showfliers=False)
plt.plot(mx1.values,'tab:blue',linewidth=2, alpha=0.5)
plt.plot(mx2.values,'tab:orange',linewidth=2, alpha=0.5)

plt.ylabel('RMSE')
plt.subplot(1,2,2)
plt.grid(True, linewidth=0.4)
mt1 = df1.groupby('grid_res')['time'].median()
mt2 = df2.groupby('grid_res')['time'].median()
sns.boxplot(x='grid_res',y='time', data=df, hue='solver', showfliers=False)
plt.plot(mt1.values,'tab:blue',linewidth=2, alpha=0.5)
plt.plot(mt2.values,'tab:orange',linewidth=2, alpha=0.5)
plt.ylabel('time')
plt.savefig('RMSE_time_AC.pdf', dpi=1000)
plt.show()

