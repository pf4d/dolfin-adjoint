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
    def __init__(self, u, refs, times, coords, verbose=False, name=None):

        combined = zip(times, refs)
        I = sum(inner(u - u_obs, u - u_obs)*ds(1)*dt[t]
                       for (t, u_obs) in combined)

        self.timeform = I
        self.coords = coords
        self.verbose = verbose
        self.name = name
        self.func = u
        self.refs = refs
        self.times = times

    def __call__(self, adjointer, timestep, dependencies, values):
        functional_value = self._substitute_form(adjointer, timestep, dependencies, values)

        if functional_value is not None:
            args = ufl.algorithms.extract_arguments(functional_value)
            if len(args) > 0:
                backend.info_red("The form passed into Functional must be rank-0 (a scalar)! You have passed in a rank-%s form." % len(args))
                raise libadjoint.exceptions.LibadjointErrorInvalidInputs

            if functional_value.coefficients()[0].element().family() is 'Real':
                solu = functional_value.coefficients()[1](self.coords)
                ref  = functional_value.coefficients()[0](self.coords)
            else:
                solu = functional_value.coefficients()[0](self.coords)
                ref  = functional_value.coefficients()[1](self.coords)
            print "da eval ", backend.assemble(functional_value)
            print "my eval ", (solu - float(ref))*(solu - float(ref))
            return (solu - float(ref))*(solu - float(ref))
        else:
            return 0.0

    def derivative(self, adjointer, variable, dependencies, values):
        functional_value = None
        for timestep in self._derivative_timesteps(adjointer, variable):
          print timestep
          functional_value = _add(functional_value,
                                  self._substitute_form(adjointer, timestep, dependencies, values))

        if functional_value is None:
            backend.info_red("Your functional is supposed to depend on %s, but does not?" % variable)
            raise libadjoint.exceptions.LibadjointErrorInvalidInputs

        d = backend.derivative(functional_value, values[dependencies.index(variable)].data)
        d = ufl.algorithms.expand_derivatives(d)
        import IPython; IPython.embed()

        if len(values)>1:
            coef = values[-2].data
            ref  = self.timeform.terms[variable.timestep+1].form.coefficients()[1]
        else:
            coef = values[0].data
            ref  = self.timeform.terms[variable.timestep].form.coefficients()[1]

        solu = coef(self.coords)
        ff = Constant(2.0*(solu - float(ref)))


        v = project(ff*PointwiseEvaluator(self.coords), coef.function_space())
        print "step", variable.timestep
        print "da", assemble(d).array()[0]
        print "my", v.vector().array()[0]

        if abs(assemble(d).array()[0]-v.vector().array()[0]) > 1e-13: import IPython; IPython.embed()

        return adjlinalg.Vector(v)

    def _substitute_form(self, adjointer, timestep, dependencies, values):
        ''' Perform the substitution of the dependencies and values
        provided. This is common to __call__ and __derivative__'''

        deps = {}
        for dep, val in zip(dependencies, values):
          deps[str(dep)] = val.data

        functional_value = None
        final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]

        # Get the necessary timestep information about the adjointer.
        # For integrals, we're integrating over /two/ timesteps.
        timestep_start, timestep_end = _time_levels(adjointer, timestep)
        point_interval = slice(timestep_start, timestep_end)

        embed()
        if timestep > 0:
          prev_start, prev_end = _time_levels(adjointer, timestep - 1)
          integral_interval = slice(prev_start, timestep_end)
        else:
          integral_interval = slice(timestep_start, timestep_end)

        for term in self.timeform.terms:
          if isinstance(term.time, slice):
            # Integral.

            def trapezoidal(interval, iteration):
              # Evaluate the trapezoidal rule for the given interval and
              # iteration. Iteration is used to select start and end of timestep.
              this_interval=_slice_intersect(interval, term.time)
              if this_interval:
                # Get adj_variables for dependencies. Time level is not yet specified.

                # Dependency replacement dictionary.
                replace={}

                term_deps = _coeffs(adjointer, term.form)
                term_vars = _vars(adjointer, term.form)

                for term_dep, term_var in zip(term_deps,term_vars):
                  this_var = self.get_vars(adjointer, timestep, term_var)[iteration]
                  replace[term_dep] = deps[str(this_var)]

                # Trapezoidal rule over given interval.
                quad_weight = 0.5*(this_interval.stop-this_interval.start)

                return backend.replace(quad_weight*term.form, replace)

            # Calculate the integral contribution from the previous time level.
            functional_value = _add(functional_value, trapezoidal(integral_interval, 0))

            # On the final occasion, also calculate the contribution from the
            # current time level.
            if adjointer.finished and timestep == adjointer.timestep_count - 1: # we're at the end, and need to add the extra terms
                                                                                # associated with that
              final_interval = slice(timestep_start, timestep_end)
              functional_value = _add(functional_value, trapezoidal(final_interval, 1))

          else:
            # Point evaluation.

            if point_interval.start < term.time < point_interval.stop:
              replace = {}

              term_deps = _coeffs(adjointer, term.form)
              term_vars = _vars(adjointer, term.form)

              for term_dep, term_var in zip(term_deps, term_vars):
                (start, end) = self.get_vars(adjointer, timestep, term_var)
                theta = 1.0 - (term.time - point_interval.start)/(point_interval.stop - point_interval.start)
                replace[term_dep] = theta*deps[str(start)] + (1-theta)*deps[str(end)]

              functional_value = _add(functional_value, backend.replace(term.form, replace))

            # Special case for evaluation at the end of time: we can't pass over to the
            # right-hand timestep, so have to do it here.
            elif (term.time == final_time or isinstance(term.time, FinishTimeConstant)) and point_interval.stop == final_time:
              replace = {}

              term_deps = _coeffs(adjointer, term.form)
              term_vars = _vars(adjointer, term.form)

              for term_dep, term_var in zip(term_deps, term_vars):
                end = self.get_vars(adjointer, timestep, term_var)[1]
                replace[term_dep] = deps[str(end)]

              functional_value = _add(functional_value, backend.replace(term.form, replace))

            # Another special case for the start of a timestep.
            elif (isinstance(term.time, StartTimeConstant) and timestep == 0) or point_interval.start == term.time:
              replace = {}

              term_deps = _coeffs(adjointer, term.form)
              term_vars = _vars(adjointer, term.form)

              for term_dep, term_var in zip(term_deps, term_vars):
                end = self.get_vars(adjointer, timestep, term_var)[0]
                replace[term_dep] = deps[str(end)]

              functional_value = _add(functional_value, backend.replace(term.form, replace))

        return functional_value

class PointwiseEvaluator(backend.Expression):
    def __init__(self, X):
        self.X = X

    def eval(self, values, x):
        values[0] = 0.0

        if all(x == self.X):
            values[0] = 1.0
