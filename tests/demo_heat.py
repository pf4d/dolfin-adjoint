from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

u_0 = Function(V)
u_1 = Function(V)

dt = Constant(0.1)

f = Expression("sin(pi*x[0])")
F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

#bc = DirichletBC(V, 1.0, "on_boundary")

a, L = lhs(F), rhs(F)

t = float(dt)
T = 1.0
n = 1

u_out = File("u.pvd", "compressed")

u_out << u_0

while( t <= T):

    solve(a == L, u_0)

    t += float(dt)
    u_out << u_0

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

print "Replay forward model"

functional=Functional(u_0*dx)

u_out = File("u_replay.pvd", "compressed")

f_direct=0.0

for i in range(adjointer.equation_count):
    (fwd_var, output) = adjointer.get_forward_solution(i)

    storage = libadjoint.MemoryStorage(output)
    storage.set_compare(tol=0.0)
    storage.set_overwrite(True)
    adjointer.record_variable(fwd_var, storage)

    u_out << output.data

# The functional is only a function of final state.
f_direct+=adjointer.evaluate_functional(functional, i)

print "Run adjoint model"


f_adj=0.0

for i in range(adjointer.equation_count)[::-1]:
    print i

    (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
    
    storage = libadjoint.MemoryStorage(output)
    adjointer.record_variable(adj_var, storage)

    if i!=0:
        # f is only the RHS from equation 1 onwards. At equation 0, the
        # RHS is the zero vector, so we don't bother. 
        f_adj+=assemble(-f*output.data*dx)

print f_adj-f_direct
