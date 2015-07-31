from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = Function(W, name="Control")

u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho * u_ * v * dx
solve(lhs(F) == rhs(F), u)

J = Functional(0.5 * inner(u, u) * dx)
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)
Jhat.hessian(rho)
