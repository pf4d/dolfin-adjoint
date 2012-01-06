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

sw.timeloop_theta(M,G,state,kelvin.params)

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

state = sw.replay(state, kelvin.params)
J = Functional(dot(state.split()[0], state.split()[0])*dx)
adj_state = sw.adjoint(state, kelvin.params, J)
