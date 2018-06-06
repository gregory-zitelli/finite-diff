'''
    PDE is
    -DtV + (r-q)*x*DxV + 0.5*sigma^2*x^2*DxxV - r*V = 0
'''

import numpy as np
import pandas as pd
from scipy.sparse import diags
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
Nt    = 10**5+1     # resolution in time
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
    Explicit method
        Simple tridiagonal forward operator A.
'''

alpha = pd.Series(0.5*np.linspace(0,Nx,Nx)**2*(sigma**2)*dt/(dx**2),index=x_grid)
beta = pd.Series(0.5*(r-q)*np.linspace(0,Nx,Nx)*dt/dx,index=x_grid)
l_e = alpha-beta ; d_e = 1-2*alpha-r*dt ; u_e = alpha+beta
# Efficiently store sparse tridiagonal matrix
explicit = diags([d_e,u_e.iloc[0:Nx-1],l_e.iloc[1:Nx] ],[0,1,-1])
V = pd.DataFrame(np.zeros((Nx,Nt)))
V.iloc[:,0] = initial_conditions

tic = time.clock()
for j in range(1,Nt):       # Approximately 302 seconds on my machine
    Vk             = V.iloc[:,j-1].copy()
    V.iloc[:,j]    = explicit*Vk                    # Forward operator
    V.iloc[0,j]    = boundary_bottom.iloc[j]        # Impose boundary conditions
    V.iloc[Nx-1,j] = boundary_top.iloc[j]           # Impose boundary conditions
    print(j)
V.index = np.linspace(Smin,Smax,Nx)
V.columns = np.linspace(0,T,Nt)
toc = time.clock()
toc-tic

import seaborn as sns
V.iloc[:,[0,20000,40000,60000,80000,100000]].plot()
plt.savefig('explicit_method.png',dpi=300,
            transparent=True,bbox_inches='tight')
plt.close("all")
