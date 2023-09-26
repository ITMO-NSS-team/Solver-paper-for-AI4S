import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import time
import pandas as pd
from scipy.integrate import quad
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


def solution_plot(model):
    plt.rc('font', size= 15 )
    solver_device('cpu')
    model = model.to('cpu')

    test_data = scipy.io.loadmat('examples/reference_sln/schrodinger_test.mat')
    data = test_data['uu'].reshape(-1, 1)
    u = np.real(data).reshape(-1)
    v = np.imag(data).reshape(-1)

    # grid_test
    x_data = torch.from_numpy(np.linspace(-5, 5, 256))
    t_data = torch.from_numpy(np.linspace(0, np.pi/2, 201))

    grid_data = torch.cartesian_prod(x_data, t_data).float()
    x_grid = np.linspace(-5, 5, 51)

    fig = plt.figure(figsize=(12,12))

    for i, time in enumerate([0.25, 0.5, 0.75, 1.]):
        t_grid = np.array([time])

        x = torch.from_numpy(x_grid)
        t = torch.from_numpy(t_grid)

        grid = torch.cartesian_prod(x, t).float()

        u_data = scipy.interpolate.griddata(grid_data, u, grid, method='cubic').reshape(-1,1)
        v_data = scipy.interpolate.griddata(grid_data, v, grid, method='cubic').reshape(-1,1)

        sol_exact = np.sqrt(u_data**2+v_data**2)

        sol = torch.sqrt(model(grid)[:,0]**2 + model(grid)[:,1]**2).detach()

    
        ax1 = fig.add_subplot(2, 2, i+1)
        if time == 1.:
            ax1.plot(x_grid, sol_exact, linewidth=4, label='exact')
            ax1.plot(x_grid, sol, '--', linewidth=4, label='numerical')
        else:
            ax1.plot(x_grid, sol_exact, linewidth=4)
            ax1.plot(x_grid, sol, '--', linewidth=4)
        ax1.set_xlabel('x')
        ax1.set_ylabel('h')
        ax1.set_title('t={}'.format(time))
        ax1.grid()
    
    ax1.legend(bbox_to_anchor=(-0.1, -0.3), loc='lower center', mode=" expand", ncol=2)
    plt.savefig('schrod_sln.pdf', dpi=1000, bbox_inches='tight')


def schrodinger_exper(grid_res, CACHE, save_sln):
    exp_dict_list = []
    solver_device("cuda")
    x_grid = np.linspace(-5,5,grid_res+1)
    t_grid = np.linspace(0,np.pi/2,grid_res+1)

    x = torch.from_numpy(x_grid)
    t = torch.from_numpy(t_grid)

    grid = torch.cartesian_prod(x, t).float()

    ## BOUNDARY AND INITIAL CONDITIONS
    fun = lambda x: 2/np.cosh(x)

    # u(x,0) = 2sech(x), v(x,0) = 0
    bnd1_real = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    bnd1_imag = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()


    # u(x,0) = 2sech(x)
    bndval1_real = fun(bnd1_real[:,0])

    #  v(x,0) = 0
    bndval1_imag = torch.from_numpy(np.zeros_like(bnd1_imag[:,0]))


    # u(-5,t) = u(5,t)
    bnd2_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    bnd2_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    bnd2_real = [bnd2_real_left,bnd2_real_right]


    # v(-5,t) = v(5,t)
    bnd2_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    bnd2_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    bnd2_imag = [bnd2_imag_left,bnd2_imag_right]


    # du/dx (-5,t) = du/dx (5,t)
    bnd3_real_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    bnd3_real_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    bnd3_real = [bnd3_real_left, bnd3_real_right]



    bop3_real = {
                'du/dx':
                    {
                        'coeff': 1,
                        'du/dx': [0],
                        'pow': 1,
                        'var': 0
                    }
    }
    # dv/dx (-5,t) = dv/dx (5,t)
    bnd3_imag_left = torch.cartesian_prod(torch.from_numpy(np.array([-5], dtype=np.float64)), t).float()
    bnd3_imag_right = torch.cartesian_prod(torch.from_numpy(np.array([5], dtype=np.float64)), t).float()
    bnd3_imag = [bnd3_imag_left,bnd3_imag_right]


    bop3_imag = {
                'dv/dx':
                    {
                        'coeff': 1,
                        'dv/dx': [0],
                        'pow': 1,
                        'var': 1
                    }
    }


    bcond_type = 'periodic'

    bconds = [[bnd1_real, bndval1_real, 0, 'dirichlet'],
            [bnd1_imag, bndval1_imag, 1, 'dirichlet'],
            [bnd2_real, 0, bcond_type],
            [bnd2_imag, 1, bcond_type],
            [bnd3_real, bop3_real, bcond_type],
            [bnd3_imag, bop3_imag, bcond_type]]

    '''
    schrodinger equation:
    i * dh/dt + 1/2 * d2h/dx2 + abs(h)**2 * h = 0 
    real part: 
    du/dt + 1/2 * d2v/dx2 + (u**2 + v**2) * v
    imag part:
    dv/dt - 1/2 * d2u/dx2 - (u**2 + v**2) * u
    u = var:0
    v = var:1
    '''

    schrodinger_eq_real = {
        'du/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            },
        '1/2*d2v/dx2':
            {
                'coeff': 1 / 2,
                'term': [0, 0],
                'pow': 1,
                'var': 1
            },
        'v * u**2':
            {
                'coeff': 1,
                'term': [[None], [None]],
                'pow': [1, 2],
                'var': [1, 0]
            },
        'v**3':
            {
                'coeff': 1,
                'term': [None],
                'pow': 3,
                'var': 1
            }

    }
    schrodinger_eq_imag = {
        'dv/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 1
            },
        '-1/2*d2u/dx2':
            {
                'coeff': - 1 / 2,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-u * v ** 2':
            {
                'coeff': -1,
                'term': [[None], [None]],
                'pow': [1, 2],
                'var': [0, 1]
            },
        '-u ** 3':
            {
                'coeff': -1,
                'term': [None],
                'pow': 3,
                'var': 0
            }

    }

    schrodinger_eq = [schrodinger_eq_real,schrodinger_eq_imag]

    model = torch.nn.Sequential(
            torch.nn.Linear(2, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 2)
        )

    start = time.time()
    equation = Equation(grid, schrodinger_eq, bconds).set_strategy('autograd')

    model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1, verbose=1, learning_rate=0.9,
                                        eps=1e-6, abs_loss=1e-6, use_cache=CACHE, cache_verbose=CACHE,
                                        save_always=CACHE, no_improvement_patience=500, print_every=None,
                                        optimizer_mode='LBFGS')
    end = time.time()

    solver_device('cpu')
    grid = grid.to(device_type())
    model = model.to(device_type())
    rmse_x_grid=np.linspace(0, 1, grid_res+1)
    rmse_t_grid=np.linspace(0, 0.2, grid_res+1)

    rmse_x = torch.from_numpy(rmse_x_grid)
    rmse_t = torch.from_numpy(rmse_t_grid)

    rmse_grid = torch.cartesian_prod(rmse_x, rmse_t).float()

    test_data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'reference_sln/schrodinger_test.mat')))
    data = test_data['uu'].reshape(-1, 1)
    u = np.real(data).reshape(-1)
    v = np.imag(data).reshape(-1)
    
    # grid_test
    x_data = torch.from_numpy(np.linspace(-5,5,256))
    t_data = torch.from_numpy(np.linspace(0,np.pi/2,201))

    grid_data = torch.cartesian_prod(x_data, t_data).float()

    u_data = scipy.interpolate.griddata(grid_data, u, rmse_grid, method='cubic').reshape(-1,1)
    v_data = scipy.interpolate.griddata(grid_data, v, rmse_grid, method='cubic').reshape(-1,1)

    u_exact = torch.hstack([torch.tensor(u_data),torch.tensor(v_data)])

    error_rmse = torch.sqrt(torch.mean((u_exact - model(rmse_grid))**2))
    
    exp_dict_list.append({'grid_res':grid_res,'time':end - start,'RMSE':error_rmse.detach().numpy(),'type':'Schrodinger_eqn','cache':CACHE})
    
    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    if save_sln:
        solution_plot(model)

    return exp_dict_list

nruns = 10

exp_dict_list=[]

CACHE=False

for grid_res in range(30,61,10):
    for i in range(nruns):
        if grid_res==50 and i==0:
            save_sln = True
        else:
            save_sln = False
        exp_dict_list.append(schrodinger_exper(grid_res,CACHE,save_sln))
   

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.to_csv('Schrod_cache={}.csv'.format(str(CACHE)))

CACHE=True

for grid_res in range(30,61,10):
    for i in range(nruns):
        exp_dict_list.append(schrodinger_exper(grid_res,CACHE, False))


exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df=pd.DataFrame(exp_dict_list_flatten)
df.to_csv('Schrod_cache={}.csv'.format(str(CACHE)))

df1 = pd.read_csv ('Schrod_cache=False.csv')
df2 = pd.read_csv ('Schrod_cache=True.csv')
df = pd.concat((df1,df2))

plt.figure(figsize=(18,8))
plt.rc('font', size= 21 ) 
plt.subplot(1,2,1)
plt.grid(True)
sns.boxplot(x='grid_res',y='RMSE', data=df, hue='cache', showfliers=False)
plt.subplot(1,2,2)
plt.grid(True, linewidth=0.4)
sns.boxplot(x='grid_res',y='time', data=df, hue='cache', showfliers=False)
plt.savefig('RMSE_time_schrod.pdf', dpi=1000)
plt.show()
