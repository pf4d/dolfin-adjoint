from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquare(16, 16)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

u_0 = Function(V)
u_1 = Function(V)

dt = Constant(0.1)

f = Expression("sin(pi*x[0])")
F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

bc = DirichletBC(V, 1.0, "on_boundary")

a, L = lhs(F), rhs(F)

u_0.adj_name = "Temperature"
u_1.adj_name = "Temperature"

t = float(dt)
T = 1.0
n = 1

while( t <= T):

    u_1.adj_timestep = n
    u_0.adj_timestep = n-1
    solve(a == L, u_1, bc)
    u_0.assign(u_1)
    t += float(dt)
    #plot(u_1, interactive=True)
    n = n + 1

adj_html("forward.html", "forward")
adj_html("adjoint.html", "forward")

adjointer.get_forward_equation(0)
