import cPickle as pickle
import hashlib
import libadjoint
import utils
from backend import Function, Constant, info_red, info_green, File
from dolfin_adjoint import drivers
from dolfin_adjoint.adjglobals import adjointer, mem_checkpoints, disk_checkpoints, adj_reset_cache
from functional import Functional
from enlisting import enlist, delist
from controls import DolfinAdjointControl, ListControl

class ReducedFunctional(object):
    ''' This class provides access to the reduced functional for given
    functional and controls. The reduced functional maps a point in control
    space to the associated functional value by implicitly solving the PDE that
    is annotated by dolfin-adjoint. The ReducedFunctional object can also
    compute functional derivatives with respect to the controls using the
    adjoint method. '''

    def __init__(self, functional, controls, scale=1.0,
                 eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 derivative_cb_pre=lambda *args: None,
                 derivative_cb_post=lambda *args: None,
                 replay_cb=lambda *args: None,
                 hessian_cb=lambda *args: None,
                 cache=None):

        #: The objective functional.
        self.functional = functional

        #: One, or a list of controls.
        self.controls = enlist(controls)

        # Check the types of the inputs
        self.__check_input_types(functional, self.controls, scale, cache)

        #: An optional scaling factor for the functional
        self.scale = scale

        #: An optional callback function that is executed before each functional
        #: evaluation.
        #: The interace must be eval_cb_pre(m) where
        #: m is the control value at which the functional is evaluated.
        self.eval_cb_pre = eval_cb_pre

        #: An optional callback function that is executed after each functional
        #: evaluation.
        #: The interace must be eval_cb_post(j, m) where j is the functional value and
        #: m is the control value at which the functional is evaluated.
        self.eval_cb_post = eval_cb_post

        #: An optional callback function that is executed before each functional
        #: gradient evaluation.
        #: The interface must be eval_cb_pre(m) where m is the control
        #: value at which the gradient is evaluated.
        self.derivative_cb_pre = derivative_cb_pre

        #: An optional callback function that is executed after each functional
        #: gradient evaluation.
        #: The interface must be eval_cb_post(j, dj, m) where j and dj are the
        #: functional and functional gradient values, and m is the control
        #: value at which the gradient is evaluated.
        self.derivative_cb_post = derivative_cb_post

        #: An optional callback function that is executed after each hessian
        #: action evaluation. The interface must be hessian_cb(j, m, mdot, h)
        #: where mdot is the direction in which the hessian action is evaluated
        #: and h the value of the hessian action.
        self.hessian_cb = hessian_cb

        #: An optional callback function that is executed after for each forward
        #: equation during a (forward) solve. The interface must be
        #: replay_cb(var, value, m) where var is the libadjoint variable
        #: containing information about the variable, value is the associated
        #: dolfin object and m is the control at which the functional is
        #: evaluated.
        self.replay_cb = replay_cb

        #: If not None, caching (memoization) will be activated. The control->ouput pairs
        #: are stored on disk in the filename given by cache.
        self.cache = cache
        if cache is not None:
            try:
                self._cache = pickle.load(open(cache, "r"))
            except IOError: # didn't exist
                self._cache = {"functional_cache": {},
                                "derivative_cache": {},
                                "hessian_cache": {}}

        #: Indicator if the user has overloaded the functional evaluation and
        #: hence re-annotates the forward model at every evaluation.
        #: By default the ReducedFunctional replays the tape for the
        #: evaluation.
        self.replays_annotation = True

        # Stores the functional value of the latest evaluation
        self.current_func_value = None

        # Set up the Hessian driver
        # Note: drivers.hessian currently only supports one control
        try:
            self.H = drivers.hessian(functional, delist(controls,
                list_type=controls), warn=False)
        except libadjoint.exceptions.LibadjointErrorNotImplemented:
            # Might fail as Hessian support is currently limited
            # to a single control
            pass

    def __check_input_types(self, functional, controls, scale, cache):

        if not isinstance(functional, Functional):
            raise TypeError("functional should be a Functional")

        for control in controls:
            if not isinstance(control, DolfinAdjointControl):
                print control.__class__
                raise TypeError("control should be a Control")

        if not isinstance(scale, float):
            raise TypeError("scale should be a float")

        if cache is not None:
            if not isinstance(cache, str):
                raise TypeError("cache should be a filename")

    def __del__(self):

        if hasattr(self, 'cache') and self.cache is not None:
            pickle.dump(self._cache, open(self.cache, "w"))

    def __call__(self, value):
        ''' Evaluates the reduced functional for the given control value. '''

        # Reset any cached data in dolfin-adjoint
        adj_reset_cache()

        #: The control values at which the reduced functional is to be evaluated.
        value = enlist(value)

        # Call callback
        self.eval_cb_pre(delist(value, list_type=self.controls))

        # Update the control values on the tape
        ListControl(self.controls).update(value)

        # Check if the result is already cached
        if self.cache:
            hash = value_hash(value)
            if hash in self._cache["functional_cache"]:
                # Found a cache
                info_green("Got a functional cache hit")
                return self._cache["functional_cache"][hash]

        # Replay the annotation and evaluate the functional
        func_value = 0.
        for i in range(adjointer.equation_count):
            (fwd_var, output) = adjointer.get_forward_solution(i)
            if isinstance(output.data, Function):
                output.data.rename(str(fwd_var), "a Function from dolfin-adjoint")

            # Call callback
            self.replay_cb(fwd_var, output.data, delist(value, list_type=self.controls))

            # Check if we checkpointing is active and if yes
            # record the exact same checkpoint variables as
            # in the initial forward run
            if adjointer.get_checkpoint_strategy() != None:
                if str(fwd_var) in mem_checkpoints:
                    storage = libadjoint.MemoryStorage(output, cs = True)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)
                if str(fwd_var) in disk_checkpoints:
                    storage = libadjoint.MemoryStorage(output)
                    adjointer.record_variable(fwd_var, storage)
                    storage = libadjoint.DiskStorage(output, cs = True)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)
                if not str(fwd_var) in mem_checkpoints and not str(fwd_var) in disk_checkpoints:
                    storage = libadjoint.MemoryStorage(output)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)

            # No checkpointing, so we record everything
            else:
                storage = libadjoint.MemoryStorage(output)
                storage.set_overwrite(True)
                adjointer.record_variable(fwd_var, storage)

            if i == adjointer.timestep_end_equation(fwd_var.timestep):
                func_value += adjointer.evaluate_functional(self.functional, fwd_var.timestep)
                if adjointer.get_checkpoint_strategy() != None:
                    adjointer.forget_forward_equation(i)

        self.current_func_value = func_value

        # Call callback
        self.eval_cb_post(self.scale * func_value, delist(value,
            list_type=self.controls))

        if self.cache:
            # Add result to cache
            info_red("Got a functional cache miss")
            self._cache["functional_cache"][hash] = self.scale*func_value

        return self.scale*func_value

    def derivative(self, forget=True, project=False):
        ''' Evaluates the derivative of the reduced functional for the most
        recently evaluated control value. '''

        # Check if we have the gradient already in the cash.
        # If so, return the cached value
        if self.cache is not None:
            hash = value_hash([x.data() for x in self.controls])
            fnspaces = [p.data().function_space() if isinstance(p.data(),
                Function) else None for p in self.controls]

            if hash in self._cache["derivative_cache"]:
                info_green("Got a derivative cache hit.")
                return cache_load(self._cache["derivative_cache"][hash], fnspaces)

        # Call callback
        values = [p.data() for p in self.controls]
        self.derivative_cb_pre(delist(values, list_type=self.controls))

        # Compute the gradient by solving the adjoint equations
        dfunc_value = drivers.compute_gradient(self.functional, self.controls, forget=forget, project=project)
        dfunc_value = enlist(dfunc_value)

        # Reset the checkpointing state in dolfin-adjoint
        adjointer.reset_revolve()

        # Apply the scaling factor
        scaled_dfunc_value = [utils.scale(df, self.scale) for df in list(dfunc_value)]

        # Call callback
        # We might have forgotten the control values already,
        # in which case we can only return Nones
        values = []
        for c in self.controls:
            try:
                values.append(p.data())
            except libadjoint.exceptions.LibadjointErrorNeedValue:
                values.append(None)
        if self.current_func_value is not None:
            self.derivative_cb_post(self.scale * self.current_func_value,
                    delist(scaled_dfunc_value, list_type=self.controls),
                    delist(values, list_type=self.controls))

        # Cache the result
        if self.cache is not None:
            info_red("Got a derivative cache miss")
            self._cache["derivative_cache"][hash] = cache_store(scaled_dfunc_value, self.cache)

        return scaled_dfunc_value

    def hessian(self, m_dot, project=False):
        ''' Evaluates the Hessian action in direction m_dot. '''

        assert(len(self.controls) == 1)

        # Check if we have the gradient already in the cash.
        # If so, return the cached value
        if self.cache is not None:
            hash = value_hash([x.data() for x in self.controls] + [m_dot])
            fnspaces = [p.data().function_space() if isinstance(p.data(),
                Function) else None for p in self.controls]

            if hash in self._cache["hessian_cache"]:
                info_green("Got a Hessian cache hit.")
                return cache_load(self._cache["hessian_cache"][hash], fnspaces)
            else:
                info_red("Got a Hessian cache miss")

        # Compute the Hessian action by solving the second order adjoint equations
        if isinstance(m_dot, list):
            assert len(m_dot) == 1
            Hm = self.H(m_dot[0], project=project)
        else:
            Hm = self.H(m_dot, project=project)

        # Apply the scaling factor
        scaled_Hm = [utils.scale(Hm, self.scale)]

        # Call callback
        control_data = [p.data() for p in self.controls]
        if self.current_func_value is not None:
            current_func_value = self.scale * self.current_func_value
        else:
            current_func_value = None

        self.hessian_cb(current_func_value,
                        delist(control_data, list_type=self.controls),
                        m_dot, scaled_Hm[0])

        # Cache the result
        if self.cache is not None:
            self._cache["hessian_cache"][hash] = cache_store(scaled_Hm, self.cache)

        return scaled_Hm


    def moola_problem(self, memoize=True):
        '''Returns a moola problem class that can be used with the moola package,
        https://github.com/funsim/moola
        '''
        import moola
        rf = self

        class Functional(moola.Functional):
            latest_eval_hash = None
            latest_eval_eval = None
            latest_eval_deriv = None

            def __call__(self, x):
                ''' Evaluates the functional for the given control value. '''

                if memoize:
                    hashx = hash(x)

                    if self.latest_eval_hash != hashx:
                        self.latest_eval_hash = hashx
                        self.latest_eval_eval = rf(x.data)
                        self.latest_eval_deriv = None
                        moola.events.increment("Functional evaluation")
                    else:
                        #print  "Using memoised functional evaluation"
                        pass

                    return self.latest_eval_eval

                else:
                    moola.events.increment("Functional evaluation")
                    return rf(x.data)


            def derivative(self, x):
                ''' Evaluates the gradient for the control values. '''

                if memoize:

                    self(x)

                    if self.latest_eval_deriv is None:
                        #print "Using memoised forward solution for gradient evaluation"
                        moola.events.increment("Derivative evaluation")
                        self.latest_eval_deriv = moola.DolfinDualVector(rf.derivative(forget=False)[0], riesz_map=x.riesz_map)

                    else:
                        #print "Using memoised gradient"
                        pass

                    return self.latest_eval_deriv

                else:
                    moola.events.increment("Derivative evaluation")
                    return moola.DolfinDualVector(rf.derivative(forget=False)[0])

            def hessian(self, x):
                ''' Evaluates the gradient for the control values. '''

                self(x)

                def moola_hessian(direction):
                    assert isinstance(direction, moola.DolfinPrimalVector)
                    moola.events.increment("Hessian evaluation")
                    hes = rf.hessian(direction.data)[0]
                    return moola.DolfinDualVector(hes)

                return moola_hessian

        functional = Functional()
        problem = moola.Problem(functional)

        return problem

def value_hash(value):
    if isinstance(value, Constant):
        return str(float(value))
    elif isinstance(value, Function):
        m = hashlib.md5()
        m.update(str(value.vector().norm("l2")) + str(value.vector().norm("l1")) + str(value.vector().norm("linf")))
        return m.hexdigest()
    elif isinstance (value, list):
        return "".join(map(value_hash, value))
    else:
        raise Exception, "Don't know how to take a hash of %s" % value

def cache_load(value, V):
    if isinstance(value, (list, tuple)):
        return [cache_load(value[i], V[i]) for i in range(len(value))]
    elif isinstance(value, float):
        return Constant(value)
    elif isinstance(value, str):
        return Function(V, value)
    return

def cache_store(value, cache):
    if isinstance(value, (list, tuple)):
        return tuple(cache_store(x, cache) for x in value)
    elif isinstance(value, Constant):
        return float(value)
    elif isinstance(value, Function):
        hash = value_hash(value)
        filename = "%s_dir/%s.xml.gz" % (cache, hash)
        File(filename) << value
        return filename
    else:
        raise Exception, "Don't know how to store %s" % value
    return
