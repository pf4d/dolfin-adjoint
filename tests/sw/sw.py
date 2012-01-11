import kelvin_new as kelvin
import sw_lib
from dolfin import *
from dolfin_adjoint import *

debugging["record_all"]=True

W=sw_lib.p1dgp2(kelvin.mesh)

state=Function(W)

state.interpolate(kelvin.InitialConditions())

kelvin.params["basename"]="p1dgp2"
kelvin.params["finish_time"]=kelvin.params["dt"]*10
kelvin.params["finish_time"]=kelvin.params["dt"]*2
kelvin.params["dump_period"]=1

M,G=sw_lib.construct_shallow_water(W,kelvin.params)

state = sw_lib.timeloop_theta(M,G,state,kelvin.params)

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")

sw_lib.replay(state, kelvin.params)
J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw_lib.adjoint(state, kelvin.params, J)

ic = Function(W)
ic.interpolate(kelvin.InitialConditions())
def J(ic):
  state = sw_lib.timeloop_theta(M, G, ic, kelvin.params, annotate=False)
  return assemble(dot(state, state)*dx)

test_initial_condition(J, ic, adj_state, seed=0.001)
