# Lotka-Volterra equations also known as predator-prey equations, describe the variation in populations
# of two species which interact via predation.
# For example, wolves (predators) and deer (prey). This is a classical model to represent the dynamic of two populations.

# Let αlpha > 0, beta > 0, delta > 0 and gamma > 0 . The system is given by

# dx/dt = x(alpha-beta*y)
# dy/dt = y(-delta+gamma*x)

# Where 'x' represents prey population and 'y' predators population. It’s a system of first-order ordinary differential equations.
import torch
import torchtext
import SALib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import fontTools
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
from tedeous.device import solver_device
from tedeous.models import mat_model


alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 2.


def Lotka_experiment(grid_res, CACHE, save_sln):
    exp_dict_list = []

    Nt = grid_res+1

    t = torch.from_numpy(np.linspace(t0, tmax, Nt))

    coord_list = [t]

    grid = grid_format_prepare(coord_list,mode='mat')

    bnd1_0 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bndval1_0 = torch.from_numpy(np.array([[x0]], dtype=np.float64))
    bnd1_1 = torch.from_numpy(np.array([[0]], dtype=np.float64)).float()
    bndval1_1  = torch.from_numpy(np.array([[y0]], dtype=np.float64))

    bconds = [[bnd1_0, bndval1_0, 0, 'dirichlet'],
              [bnd1_1, bndval1_1, 1, 'dirichlet']]

    #equation system
    # eq1: dx/dt = x(alpha-beta*y)
    # eq2: dy/dt = y(-delta+gamma*x)

    # x var: 0
    # y var:1

    eq1 = {
        'dx/dt':{
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': [0]
        },
        '-x*alpha':{
            'coeff': -alpha,
            'term': [None],
            'pow': 1,
            'var': [0]
        },
        '+beta*x*y':{
            'coeff': beta,
            'term': [[None], [None]],
            'pow': [1, 1],
            'var': [0, 1]
        }
    }

    eq2 = {
        'dy/dt':{
            'coeff': 1,
            'term': [0],
            'pow': 1,
            'var': [1]
        },
        '+y*delta':{
            'coeff': delta,
            'term': [None],
            'pow': 1,
            'var': [1]
        },
        '-gamma*x*y':{
            'coeff': -gamma,
            'term': [[None], [None]],
            'pow': [1, 1],
            'var': [0, 1]
        }
    }

    Lotka = [eq1, eq2]

    model = mat_model(grid, Lotka)

    equation = Equation(grid, Lotka, bconds).set_strategy('mat')

    start = time.time()

    model = Solver(grid, equation, model, 'mat').solve(lambda_bound=100, derivative_points=3, gamma=0.9, lr_decay=400,
                                         verbose=True, learning_rate=1, eps=1e-6, use_cache=CACHE, cache_verbose=CACHE,
                                         print_every=None, patience=3, optimizer_mode='LBFGS')

    end = time.time()
    
    def exact():
        # scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

        def deriv(X, t, alpha, beta, delta, gamma):
            x, y = X
            dotx = x * (alpha - beta * y)
            doty = y * (-delta + gamma * x)
            return np.array([dotx, doty])

        t = np.linspace(0.,tmax, Nt)

        X0 = [x0, y0]
        res = integrate.odeint(deriv, X0, t, args = (alpha, beta, delta, gamma))
        x, y = res.T
        return np.array([x.reshape(-1), y.reshape(-1)])

    u_exact = exact()

    u_exact=torch.from_numpy(u_exact)

    error_rmse=torch.sqrt(torch.mean((u_exact-model)**2))
    
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'Lotka_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    if save_sln:
        plt.figure()
        plt.grid()
        plt.plot(t, u_exact[0].detach().numpy().reshape(-1), '+', label = 'x_odeint')
        plt.plot(t, u_exact[1].detach().numpy().reshape(-1), '*', label = "y_odeint")
        plt.plot(t, model[0].detach().numpy().reshape(-1), label='x_tedeous')
        plt.plot(t, model[1].detach().numpy().reshape(-1), label='y_tedeous')
        plt.xlabel('Time, t')
        plt.ylabel('Population')
        plt.legend(loc='upper right')
        plt.savefig('LV_sln.pdf', dpi=1000)

    return exp_dict_list

nruns = 10

exp_dict_list=[]

CACHE=False

for grid_res in range(200, 401, 50):
    for i in range(nruns):
        if grid_res == 400 and i == 0:
            save_sln = True
        else:
            save_sln = False
        exp_dict_list.append(Lotka_experiment(grid_res,CACHE, save_sln))


exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
df.to_csv('LV.csv')

df = pd.read_csv ('LV.csv')
plt.figure(figsize=(18,8))
plt.rc('font', size= 21 )
plt.subplot(1,2,1)
plt.grid(True)
mx1 = df.groupby('grid_res')['RMSE'].median()
sns.boxplot(x='grid_res',y='RMSE', data=df, showfliers=False)
plt.plot(mx1.values,'tab:blue',linewidth=2, alpha=0.5)
plt.ylabel('rmse')
plt.subplot(1,2,2)
plt.grid(True, linewidth=0.4)
mt1 = df.groupby('grid_res')['time'].median()
sns.boxplot(x='grid_res',y='time', data=df, showfliers=False)
plt.plot(mt1.values,'tab:blue',linewidth=2, alpha=0.5)
plt.ylabel('time')
plt.savefig('RMSE_time_LV.pdf', dpi=1000)
plt.show()

