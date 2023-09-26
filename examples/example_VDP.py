
import torch
import torchtext
import SALib
import numpy as np
import pandas as pd
import seaborn as sns
import fontTools
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver, grid_format_prepare
from tedeous.solution import Solution
from tedeous.device import solver_device, device_type
from tedeous.models import FourierNN

eps = 0.2
t0 = 0.
tmax = 16.


def Van_DP_experiment(grid_res, CACHE, save_plot):
    exp_dict_list = []

    Nt = grid_res+1

    solver_device('cuda')

    t = torch.from_numpy(np.linspace(t0, tmax, Nt))

    grid = t.reshape(-1, 1).float()

    bnd1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bndval1 = torch.from_numpy(np.array([[np.sqrt(3)/2]], dtype=np.float64))
    bnd2 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bop2 = {
       'du/dt':{
            'coeff': 1,
            'term': [0],
            'pow': 1
        }
    }

    bndval2  = torch.from_numpy(np.array([[1/2]], dtype=np.float64))

    bconds = [[bnd1, bndval1, 'dirichlet'],
              [bnd2, bop2, bndval2, 'operator']]

    vdp = {
        'd2u/dt2':{
            'coeff': 1,
            'term': [0,0],
            'pow': 1,
            'var': [0]
        },
        '+eps*u**2*du/dt':{
            'coeff': eps,
            'term': [[None],[0]],
            'pow': [2, 1],
            'var': [0, 0]
        },
        '-eps*du/dt':{
            'coeff': -eps,
            'term': [0],
            'pow': 1,
            'var': [0]
        },
        '+u':{
            'coeff': 1.,
            'term': [None],
            'pow': 1,
            'var': [0]
        }
    }

    model = FourierNN([512, 512, 512, 512, 1], [10], [2])

    equation = Equation(grid, vdp, bconds).set_strategy('autograd')

    img_dir = os.path.join(os.path.dirname( __file__ ), 'img_VDP_paper')

    start = time.time()

    model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=100, print_every=None,
                                         verbose=True, learning_rate=1e-3, eps=1e-6, tmax=5e6,
                                         use_cache=CACHE, save_always=CACHE, patience=2, abs_loss=1e-7,
                                         model_randomize_parameter=1e-5, image_save_dir=img_dir)

    end = time.time()

    def exact():
        # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

        def deriv(X, t, eps):
            u, v = X
            dotu = v
            dotv = -eps*(u**2-1)*v-u
            return np.array([dotu, dotv])

        t = np.linspace(0.,16, Nt)

        X0 = [np.sqrt(3)/2, 1/2]
        res = integrate.odeint(deriv, X0, t, args = (eps,))
        u, v = res.T
        return np.array(u).reshape(-1), np.array(v).reshape(-1)

    u_exact, v_exact = exact()

    device = device_type()

    u_exact = torch.from_numpy(u_exact).to(device)

    model = model.to(device)

    grid = grid.to(device)

    error_rmse = torch.sqrt(torch.mean((u_exact-model(grid).reshape(-1))**2))
    
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().cpu().numpy(),'type':'VDP_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    if save_plot:
        plt.figure()
        plt.grid()
        plt.plot(t, u_exact.detach().cpu().numpy().reshape(-1), '+', label = 'u_odeint')
        plt.plot(t, model(grid).detach().cpu().numpy().reshape(-1), label = "u_tedeous")
        plt.xlabel('t')
        plt.ylabel('u')
        plt.legend(loc='upper right')
        plt.savefig('VDP_solution.pdf', dpi=1000)

    return exp_dict_list

nruns = 10

exp_dict_list=[]

CACHE=False

for grid_res in range(40, 121, 20):
    for i in range(nruns):
        if grid_res==120 and i==0:
            save_plot=True
        else:
            save_plot=False
        exp_dict_list.append(Van_DP_experiment(grid_res, CACHE, save_plot))

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)

df.to_csv('VDP.csv')
df = pd.read_csv('VDP.csv')
plt.figure(figsize=(18, 8))
plt.rc('font', size= 21)
plt.subplot(1, 2, 1)
plt.grid(True)
sns.boxplot(x='grid_res', y='RMSE', data=df, hue='type', showfliers=False)
plt.ylabel('RMSE')
plt.subplot(1, 2, 2)
plt.grid(True, linewidth=0.4)
sns.boxplot(x='grid_res', y='time', data=df, hue='type', showfliers=False)
plt.ylabel('time')
plt.savefig('RMSE_time_VDP.pdf', dpi=1000)
plt.show()