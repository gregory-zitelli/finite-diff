'''
    PDE is
    -DtV + (r-q)*x*DxV + 0.5*sigma^2*x^2*DxxV - r*V = 0
'''

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.linalg import solve_banded
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
    Set up parameters.
'''

r     = 0.03        # risk-free rate
q     = 0           # dividend yield
sigma = 0.2         # volatility of underlying
T     = 10          # horizon (right edge)
Smin  = 0           # top edge
Smax  = 200         # bottom edge
Nt    = 10**2+1     # resolution in time
Nx    = 200+1       # resolution in space
dt    = T/(Nt-1)
dx    = (Smax-Smin)/(Nx-1)
# Call with strike K
K     = 105
# Set up grids
x_grid = np.linspace(Smin,Smax,Nx)
t_grid = np.linspace(0,T,Nt)
# Set up initial conditions
initial_conditions = pd.Series(x_grid).apply(lambda z:max(0,z-K))
# Set up Dirichlet boundary conditions
boundary_bottom    = pd.Series(np.zeros(Nt),index=t_grid)
boundary_top       = pd.Series(max(0,Smax-K)*np.exp(-r*t_grid),index=t_grid)

'''
    Implicit method
        Solve linear equation Bv_k+1 = v_k + b
        for the implicit matrix B.
'''

alpha = pd.Series(0.5*np.linspace(0,Nx,Nx)**2*(sigma**2)*dt/(dx**2),index=x_grid)
beta = pd.Series(0.5*(r-q)*np.linspace(0,Nx,Nx)*dt/dx,index=x_grid)
l_i = -alpha+beta ; d_i = 1+r*dt+2*alpha ; u_i = -alpha-beta

V = pd.DataFrame(np.zeros((Nx,Nt)))
V.iloc[:,0] = initial_conditions

for j in range(1,Nt):       # Approximately 0.1449 seconds on my machine
    Vk             = V.iloc[:,j-1].copy()
    Vk.iloc[0]    += -l_i[0]*boundary_bottom.iloc[j]    # Boundary conditions
    Vk.iloc[Nx-1] += -u_i[Nx-1]*boundary_top.iloc[j]    # Boundary conditions
    V.iloc[:,j]    = solve_banded((1, 1),               # Solve implicit system
                                  np.array([u_i.shift(1).fillna(0).tolist(),
                                            d_i.tolist(),
                                            l_i.shift(-1).fillna(0).tolist()]),Vk)
V.index = np.linspace(Smin,Smax,Nx)
V.columns = np.linspace(0,T,Nt)

import seaborn as sns
V.iloc[:,[0,20,40,60,80,100]].plot()
plt.savefig('implicit_method',dpi=300,
            transparent=True,bbox_inches='tight')
plt.close("all")
