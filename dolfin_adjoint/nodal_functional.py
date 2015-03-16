import libadjoint
from ufl import *
import ufl.algorithms
import backend
import hashlib

import solving
import functional
from functional import _time_levels, _add, _coeffs, _vars
import adjglobals
import adjlinalg
from timeforms import NoTime, StartTimeConstant, FinishTimeConstant, dt, FINISH_TIME

from IPython import embed

class NodalFunctional(functional.Functional):
    def __init__(self, timeform, u, refs, times, coords, verbose=False, name=None):

#        combined = zip(times, refs)
#        I = sum(inner(u - u_obs, u - u_obs)*ds(1)*dt[t]
#                       for (t, u_obs) in combined)

        self.timeform = timeform
        self.coords = coords
        self.verbose = verbose
        self.name = name
        self.func = u
        self.refs = refs
        self.times = times

    def __call__(self, adjointer, timestep, dependencies, values):
        print "eval ", timestep, " times ", _time_levels(adjointer, timestep)
        functional_value = self._substitute_form(adjointer, timestep, dependencies, values)

        if functional_value is not None:
            args = ufl.algorithms.extract_arguments(functional_value)
            if len(args) > 0:
                backend.info_red("The form passed into Functional must be rank-0 (a scalar)! You have passed in a rank-%s form." % len(args))
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs

#            embed()

            toi = _time_levels(adjointer, timestep)[0]
            solu = values[-1].data(self.coords)
            ref  = self.refs[self.times.index(toi)]
            my = (solu - float(ref))*(solu - float(ref))

            if len(values) > 1:
                toi = self.times[-1]
                solu = values[0].data(self.coords)
                ref  = self.refs[self.times.index(toi)]
                my += (solu - float(ref))*(solu - float(ref))

            da = backend.assemble(functional_value)

            print "da eval ", da
            print "my eval ", my
            if abs(da - my) > 1e-13: embed()
            return my
        else:
            return 0.0

    def derivative(self, adjointer, variable, dependencies, values):
        functional_value = None
        for timestep in self._derivative_timesteps(adjointer, variable):
          print "derive ", timestep, " times ", _time_levels(adjointer, timestep)
          functional_value = _add(functional_value,
                                  self._substitute_form(adjointer, timestep, dependencies, values))

        if functional_value is None:
            backend.info_red("Your functional is supposed to depend on %s, but does not?" % variable)
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs

        d = backend.derivative(functional_value, values[dependencies.index(variable)].data)
        d = ufl.algorithms.expand_derivatives(d)

        tsoi = self._derivative_timesteps(adjointer, variable)[0]
        toi = _time_levels(adjointer, tsoi)[-1]
        if toi in self.times:
            "showtime"
            if len(values) > 1: # occurs for timestep 0
                coef = values[-2].data
            else :
                coef = values[0].data
                toi = _time_levels(adjointer, tsoi)[0]

            ref  = self.refs[self.times.index(toi)]
            solu = coef(self.coords)
            ff = backend.Constant(2.0*(solu - float(ref)))
        else:
            ff = 0.0

        v = backend.project(ff*PointwiseEvaluator(self.coords), coef.function_space())
        da = backend.assemble(d).array()[0]
        my = v.vector().array()[0]
        print "step", variable.timestep
        print "da", da
        print "my", my
#        embed()
        if abs(da - my) > 1e-13: embed()

        if abs(backend.assemble(d).array()[0]-v.vector().array()[0]) > 1e-13: import IPython; IPython.embed()

        return adjlinalg.Vector(v)

class PointwiseEvaluator(backend.Expression):
    def __init__(self, X):
        self.X = X

    def eval(self, values, x):
        values[0] = 0.0

        if all(x == self.X):
            values[0] = 1.0
