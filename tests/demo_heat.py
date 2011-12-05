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

u_out = File("u.pvd", "compressed")

u_out << u_0


while( t <= T):

    u_1.adj_timestep = n
    u_0.adj_timestep = n-1
    
    solve(a == L, u_1, bc)
    u_0.assign(u_1)
    t += float(dt)
    #plot(u_1, interactive=True)
    n = n + 1

    u_out << u_0

adj_html("forward.html", "forward")
adj_html("adjoint.html", "forward")

print "Replay forward model"

functional=Functional(u_1*u_1*dx)

u_out = File("u_replay.pvd", "compressed")

for i in range(adjointer.equation_count):
    (fwd_var, output) = adjointer.get_forward_solution(i)

    adjointer.record_variable(fwd_var, libadjoint.MemoryStorage(output))

    u_out << output.data

(fwd_var, output) = adjointer.get_adjoint_solution(10, functional)
