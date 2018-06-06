using DataFrames

# Set up parameters.
r     = 0.03;       # risk-free rate
q     = 0;          # dividend yield
sigma = 0.2;        # volatility of underlying
T     = 10;         # horizon (right edge)
Smin  = 0;          # top edge
Smax  = 200;        # bottom edge
Nt    = 10^2+1;     # resolution in time
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

# Implicit method
#   Solve linear equation Bv_k+1 = v_k + b
#   for the implicit matrix B.
a_i = 0.5*[i^2 for i in linspace(0,Nx-1,Nx)]*(sigma^2)*dt/(dx^2);
b_i = 0.5*(r-q)*[i for i in linspace(0,Nx-1,Nx)]*dt/dx;
l_i = -a_i+b_i; d_i = 1+r*dt+2*a_i; u_i = -a_i-b_i;
implicit = spdiagm(0 => d_i, -1 => l_i[2:Nx], 1 => u_i[1:Nx-1]);
V = zeros((Nx,Nt));
V[:,1] = initial_conditions;

# @time for j in 2:Nt
for j in 2:Nt
    Vk      = copy(V[:,j-1]);
    Vk[1]  += -l_i[1]*boundary_bottom[j];   # Boundary conditions
    Vk[Nx] += -u_i[Nx]*boundary_top[j];     # Boundary conditions
    V[:,j]  = implicit\ Vk;
end
# 0.558045 seconds (258.62 k allocations: 36.297 MiB, 2.43% gc time)

using Seaborn
ioff()
Seaborn.plot(V[:,[1,21,41,61,81,101]])
Seaborn.savefig("implicit_method.png",dpi=300,transparent=true,bbox_inches="tight")
