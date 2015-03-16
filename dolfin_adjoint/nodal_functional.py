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
    def __init__(self, u, refs, coords, times=None, f_ind=None, timeform=False, verbose=False, name=None):
        if times is None:
            times = ["FINISH_TIME"]
        # we prepare a ghost timeform. Only the time instant is important
        if not timeform: self.timeform = sum(u*dx*dt[t] for t in times)
        else: self.timeform = timeform

        self.coords = coords
        self.verbose = verbose
        self.name = name
        self.func = u
        self.refs = refs
        self.times = times
        self.i = f_ind

    def __call__(self, adjointer, timestep, dependencies, values):
        print "eval ", len(values)
        toi = _time_levels(adjointer, timestep)[0]
        if len(values) > 0:
            if timestep is adjointer.timestep_count -1:
                # add final contribution
                if self.i is None: solu = values[0].data(self.coords)
                ref  = self.refs[self.times.index(self.times[-1])]
                my = (solu - float(ref))*(solu - float(ref))

                # if necessary, add one but last contribution
                if toi in self.times and len(values) > 0:
                    if self.i is None: solu = values[-1].data(self.coords)
                    ref  = self.refs[self.times.index(toi)]
                    my += (solu - float(ref))*(solu - float(ref))
            else:
                if self.i is None: solu = values[-1].data(self.coords)
                ref  = self.refs[self.times.index(toi)]
                my = (solu - float(ref))*(solu - float(ref))
        else:
            my = 0.0

        print "my eval ", my
        print "eval ", timestep, " times ", _time_levels(adjointer, timestep)
        functional_value = self._substitute_form(adjointer, timestep, dependencies, values)

        if functional_value is not None:
            args = ufl.algorithms.extract_arguments(functional_value)
            if len(args) > 0:
                backend.info_red("The form passed into Functional must be rank-0 (a scalar)! You have passed in a rank-%s form." % len(args))
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs

            da = backend.assemble(functional_value)
        else:
            da = 0.0

        print "da eval ", da
        if abs(da - my) > 1e-13: embed()
        return my

    def derivative(self, adjointer, variable, dependencies, values):
        #transate finish_time: UGLY!!
        if "FINISH_TIME" in self.times:
            final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]
            self.times[self.times.index("FINISH_TIME")] = final_time

        print "derive ", variable.timestep, " num values ", len(values)
        timesteps = self._derivative_timesteps(adjointer, variable)
        if len(timesteps) is 1: # only occurs at atart and finish time
            tsoi = timesteps[-1]
            if tsoi is 0: toi = _time_levels(adjointer, tsoi)[0]; ind = -1
            else: toi = _time_levels(adjointer, tsoi)[-1]; ind = 0
        else:
            if len(values) is 1: # one value (easy)
                tsoi = timesteps[-1]
                toi = _time_levels(adjointer, tsoi)[0]
                ind = 0
            elif len(values) is 2: # two values (hard)
                tsoi = timesteps[-1]
                toi = _time_levels(adjointer, tsoi)[0]
                if _time_levels(adjointer, tsoi)[1] in self.times: ind = 0
                else: ind = 1
            else: # three values (easy)
                tsoi = timesteps[1]
                toi = _time_levels(adjointer, tsoi)[0]
                ind = 1
        coef = values[ind].data
        ref  = self.refs[self.times.index(toi)]
        if self.i is None: solu = coef(self.coords)
        else: solu = coef[self.i](self.coords)
        ff = backend.Constant(2.0*(solu - float(ref)))

        v = backend.project(ff*PointwiseEvaluator(self.coords), coef.function_space())
        my = v.vector().array()[0]

        print "my", my

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
        da = backend.assemble(d).array()[0]
        print "da", da
        print "tsoi", tsoi
        print "toi", toi
        if abs(da - my) > 1e-13: embed()

        return adjlinalg.Vector(v)

class PointwiseEvaluator(backend.Expression):
    def __init__(self, X):
        self.X = X

    def eval(self, values, x):
        values[0] = 0.0

        if all(x == self.X):
            values[0] = 1.0
