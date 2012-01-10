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

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")

sw.replay(state, kelvin.params)
J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw.adjoint(state, kelvin.params, J)

ic = Function(W)
ic.interpolate(kelvin.InitialConditions())
def J(ic):
  state = sw.timeloop_theta(M, G, ic, kelvin.params, annotate=False)
  return assemble(dot(state, state)*dx)

test_initial_condition(J, ic, adj_state, seed=0.001)
