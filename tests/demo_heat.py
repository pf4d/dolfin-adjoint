from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True

f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)")
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)

def run_forward(initial_condition=None, annotate=True, dump=True):
  u = TrialFunction(V)
  v = TestFunction(V)

  u_0 = Function(V)
  if initial_condition is not None:
    u_0.assign(initial_condition)

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

      solve(a == L, u_0, bc, annotate=annotate)
      #solve(a == L, u_0, annotate=annotate)

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
functional=Functional(u_0*u_0*dx)
f_direct = adjointer.evaluate_functional(functional, adjointer.equation_count-1)

print "Running adjoint model ..."

def run_adjoint():
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

  return (f_adj, output.data)

(f_adj, final_adjoint) = run_adjoint()

#print "Evaluating functional only using adjoint ... "
#print "Difference between the functional evaluatons (should be 0): ", f_adj-f_direct

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders


def test_ic_gradient(final_adjoint):
  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  print "Running Taylor remainder convergence analysis ... "
  import random
  import numpy

  # Randomise the perturbation direction:
  perturbation_direction = Function(V)
  vec = perturbation_direction.vector()
  for i in range(len(vec)):
    vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [10.0/(2**i) for i in range(5)]:
    perturbation = Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_u0 = run_forward(initial_condition=perturbation, annotate=False, dump=False)
    functional_values.append(assemble(perturbed_u0*perturbed_u0*dx))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]
  no_gradient_conv = convergence_order(no_gradient)

  print "Taylor remainder without adjoint information: ", no_gradient
  print "Convergence orders for Taylor remainder without adjoint information (should all be 1): ", no_gradient_conv

  adjoint_vector = numpy.array(final_adjoint.vector())

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - f_direct - numpy.dot(adjoint_vector, numpy.array(perturbations[i].vector())))
    with_gradient.append(remainder)

  print "Taylor remainder with adjoint information: ", with_gradient
  print "Convergence orders for Taylor remainder wth adjoint information (should all be 2): ", convergence_order(with_gradient)

test_ic_gradient(final_adjoint)
