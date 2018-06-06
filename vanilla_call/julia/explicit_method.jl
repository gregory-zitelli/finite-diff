using DataFrames

# Set up parameters.
r     = 0.03;       # risk-free rate
q     = 0;          # dividend yield
sigma = 0.2;        # volatility of underlying
T     = 10;         # horizon (right edge)
Smin  = 0;          # top edge
Smax  = 200;        # bottom edge
Nt    = 10^5+1;     # resolution in time
Nx    = 200+1;      # resolution in space
dt    = T/(Nt-1);
dx    = (Smax-Smin)/(Nx-1);
# Call with strike K
K     = 105;
# Set up grids
x_grid = linspace(Smin,Smax,Nx);
t_grid = linspace(0,T,Nt);
# Set up initial conditions
initial_conditions = map(z -> max(0,z-K),x_grid);
# Set up Dirichlet boundary conditions
boundary_bottom    = zeros(Nt);
boundary_top       = max(0,Smax-K)*map(exp,-r*t_grid);

# Explicit method
#   Simple tridiagonal forward operator A.
a_e = 0.5*[i^2 for i in linspace(0,Nx-1,Nx)]*(sigma^2)*dt/(dx^2);
b_e = 0.5*(r-q)*[i for i in linspace(0,Nx-1,Nx)]*dt/dx;
l_e = a_e-b_e; d_e = 1-2*a_e-r*dt; u_e=a_e+b_e;
# Efficiently store sparse tridiagonal matrix
explicit = spdiagm(0 => d_e, -1 => l_e[2:Nx], 1 => u_e[1:Nx-1]);
V = zeros((Nx,Nt));
V[:,1] = initial_conditions;

# @time for j in 2:Nt
for j in 2:Nt
    Vk          = copy(V[:,j-1]);
    V[:,j]      = copy(explicit*Vk);            # Forward operator
    V[1,j]     += l_e[1]*boundary_bottom[j];    # Boundary conditions
    V[Nx,j]    += u_e[Nx]*boundary_top[j];      # Boundary conditions
end
# 0.618032 seconds (2.03 M allocations: 717.108 MiB, 43.11% gc time)

using Seaborn
ioff()
Seaborn.plot(V[:,[1,20001,40001,60001,80001,100001]])
Seaborn.savefig("explicit_method.png",dpi=300,transparent=true,bbox_inches="tight")
