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
    Crank–Nicolson method
        Solve linear equation (B+I)v_k+1 = (A_I)v_k + b
        for the explicit matrix A and implicit matrix B.
'''

alpha = pd.Series(0.5*np.linspace(0,Nx,Nx)**2*(sigma**2)*dt/(dx**2),index=x_grid)
beta = pd.Series(0.5*(r-q)*np.linspace(0,Nx,Nx)*dt/dx,index=x_grid)
l_e = alpha-beta ; d_e = 1-r*dt-2*alpha ; u_e = alpha+beta
# Efficiently store sparse tridiagonal matrix
explicit = diags([d_e,u_e.iloc[0:Nx-1],l_e.iloc[1:Nx] ],[0,1,-1])
l_i = -alpha+beta ; d_i = 1+r*dt+2*alpha ; u_i = -alpha-beta

V = pd.DataFrame(np.zeros((Nx,Nt)))
V.iloc[:,0] = initial_conditions
for j in range(1,Nt):       # Approximately 0.1696 seconds on my machine
    Vk             = V.iloc[:,j-1].copy()
    Vk             = explicit*Vk + Vk                           # Forward operator
    Vk.iloc[0]    += (l_e[0]-l_i[0])*boundary_bottom.iloc[j]    # Minor boundary issues
    Vk.iloc[Nx-1] += (u_e[Nx-1]-u_i[Nx-1])*boundary_top.iloc[j] # Minor boundary issues
    V.iloc[:,j]    = solve_banded((1, 1),                       # Solve implicit system
                                  np.array([u_i.shift(1).fillna(0).tolist(),
                                            (d_i+1).tolist(),
                                            l_i.shift(-1).fillna(0).tolist()]),Vk)
V.index = np.linspace(Smin,Smax,Nx)
V.columns = np.linspace(0,T,Nt)

import seaborn as sns
V.iloc[:,[0,20,40,60,80,100]].plot()
plt.savefig('crank_nicolson_method',dpi=300,
            transparent=True,bbox_inches='tight')
plt.close("all")
