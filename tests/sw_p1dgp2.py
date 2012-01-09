import kelvin_new as kelvin
import sw
from dolfin import *
from dolfin_adjoint import *

debugging["record_all"]=True

W=sw.p1dgp2(kelvin.mesh)

state=Function(W)

state.interpolate(kelvin.InitialConditions())

kelvin.params["basename"]="p1dgp2"
kelvin.params["finish_time"]=kelvin.params["dt"]*10
kelvin.params["finish_time"]=kelvin.params["dt"]*2
kelvin.params["dump_period"]=1

M,G=sw.construct_shallow_water(W,kelvin.params)

state = sw.timeloop_theta(M,G,state,kelvin.params)

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

sw.replay(state, kelvin.params)
J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw.adjoint(state, kelvin.params, J)

def test_ic_gradient(final_adjoint):
  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  print "Running Taylor remainder convergence analysis ... "
  import random
  import numpy

  # Randomise the perturbation direction:
  perturbation_direction = Function(W)
  vec = perturbation_direction.vector()
  for i in range(len(vec)):
    vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [0.001/(2**i) for i in range(5)]:
    perturbation = Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = Function(W)
    perturbed_ic.interpolate(kelvin.InitialConditions())
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    perturbed_state = sw.timeloop_theta(M, G, perturbed_ic, kelvin.params, annotate=False)
    functional_values.append(assemble(dot(perturbed_state, perturbed_state)*dx))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  print "Taylor remainder without adjoint information: ", no_gradient
  print "Convergence orders for Taylor remainder without adjoint information (should all be 1): ", convergence_order(no_gradient)

  adjoint_vector = numpy.array(final_adjoint.vector())

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - f_direct - numpy.dot(adjoint_vector, numpy.array(perturbations[i].vector())))
    with_gradient.append(remainder)

  print "Taylor remainder with adjoint information: ", with_gradient
  print "Convergence orders for Taylor remainder wth adjoint information (should all be 2): ", convergence_order(with_gradient)

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders

test_ic_gradient(adj_state)
