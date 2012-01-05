from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True

f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)")

def run_forward(initial_condition="0.0", annotate=True, dump=True):
  mesh = UnitSquare(4, 4)
  V = FunctionSpace(mesh, "CG", 1)

  u = TrialFunction(V)
  v = TestFunction(V)

  u_0 = Function(V)
  ic_expression = Expression(initial_condition)
  u_0.interpolate(ic_expression)

  u_1 = Function(V)

  dt = Constant(0.1)

  F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

  bc = DirichletBC(V, 1.0, "on_boundary")

  a, L = lhs(F), rhs(F)

  t = float(dt)
  T = 1.0
  n = 1

  if dump:
    u_out = File("u.pvd", "compressed")
    u_out << u_0

  while t <= T:

      #solve(a == L, u_0, bc, annotate=annotate)
      solve(a == L, u_0, annotate=annotate)

      t += float(dt)
      if dump:
        u_out << u_0

  return u_0


u_0 = run_forward()

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

def run_replay():
  print "Replaying forward model (will error if differs from correct solution) ..."


  u_out = File("u_replay.pvd", "compressed")

  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=0.0)
      storage.set_overwrite(True)
      adjointer.record_variable(fwd_var, storage)

      u_out << output.data

run_replay()

# The functional is only a function of final state.
functional=Functional(u_0*dx)
f_direct = adjointer.evaluate_functional(functional, adjointer.equation_count-1)

print "Running adjoint model ..."

z_out = File("adjoint.pvd", "compressed")
f_adj=0.0

for i in range(adjointer.equation_count)[::-1]:

    (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
    
    storage = libadjoint.MemoryStorage(output)
    adjointer.record_variable(adj_var, storage)

    z_out << output.data

    if i!=0:
        # f is only the RHS from equation 1 onwards. At equation 0, the
        # RHS is the zero vector, so we don't bother. 
        f_adj+=assemble(-f*output.data*dx)

print "Evaluating functional only using adjoint ... "
print "Difference between the functional evaluatons: ", f_adj-f_direct
