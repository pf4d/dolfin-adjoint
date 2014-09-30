import numpy as np
from backend import cpp, info, info_red, Constant, Function, TestFunction, TrialFunction, assemble, inner, dx, as_backend_type, info_red, MPI
from dolfin_adjoint import constant, utils 
from dolfin_adjoint.adjglobals import adjointer, adj_reset_cache
from reduced_functional import ReducedFunctional
from utils import gather
from functools import partial
import misc

class ReducedFunctionalNumPy(ReducedFunctional):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the parameter value which 
        implicitly solves the recorded PDE. '''

    def __init__(self, rf):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            This "NumPy version" of the reduced functional is created from an existing ReducedFunctional object:
              rf_np = ReducedFunctionalNumPy(rf = rf)
        '''
        super(ReducedFunctionalNumPy, self).__init__(rf.functional, rf.parameter, scale=rf.scale, 
                                                     eval_cb=rf.eval_cb, derivative_cb=rf.derivative_cb, 
                                                     replay_cb=rf.replay_cb, hessian_cb=rf.hessian_cb, 
                                                     cache=rf.cache)
        self.current_func_value = rf.current_func_value

        self.__base_call__ = rf.__call__
        self.__base_derivative__ = rf.derivative
        self.__base_hessian__ = rf.hessian

        self.rf = rf


    def __call__(self, m_array):
        ''' An implementation of the reduced functional evaluation
            that accepts the parameter as an array of scalars '''

        # In case the annotation is not reused, we need to reset any prior annotation of the adjointer before reruning the forward model.
        if not self.replays_annotation:
            solving.adj_reset()

        # We move in parameter space, so we also need to reset the factorisation cache
        adj_reset_cache()

        # Now its time to update the parameter values using the given array  
        m = self.rf.parameter.__class__([p.data() for p in self.parameter])
        self.set_local(m, m_array)

        return self.__base_call__(m)

    def set_local(self, m, m_array):
        set_local(m, m_array)

    def get_global(self, m):
        return get_global(m)

    def derivative(self, m_array=None, taylor_test=False, seed=0.001, forget=True, project=False):
        ''' An implementation of the reduced functional derivative evaluation 
            that accepts the parameter as an array of scalars. If no parameter values are given,
            the result is derivative at the last forward run.
            If taylor_test = True, the derivative is automatically verified 
            by the Taylor remainder convergence test. The perturbation direction 
            is random and the perturbation size can be controlled with the seed argument.
            '''

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        m = [p.data() for p in self.parameter]
        if m_array is not None and (m_array != self.get_global(m)).any():
            info_red("Rerunning forward model before computing derivative")
            self(m_array) 

        dJdm = self.__base_derivative__(forget=forget, project=project) 
        if project:
            dJdm_global = self.get_global(dJdm)
        else:
            dJdm_global = get_global(dJdm)

        # Perform the gradient test
        if taylor_test:
            minconv = utils.test_gradient_array(self.__call__, self.scale * dJdm_global, m_array, 
                                                seed = seed) 
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test successful.")
            self(m_array) 

        return dJdm_global 

    def hessian(self, m_array, m_dot_array):
        ''' An implementation of the reduced functional hessian action evaluation 
            that accepts the parameter as an array of scalars. If m_array is None,
            the Hessian action at the latest forward run is returned. ''' 

        if not hasattr(self, "H"):
            raise NotImplementedError, "Hessian computation not supported."

        m = [p.data() for p in self.parameter]
        if m_array is not None:
            # In case the parameter values have changed since the last forward run, 
            # we first need to rerun the forward model with the new parameters to have the 
            # correct forward solutions
            if (m_array != self.get_global(m)).any():
                self(m_array) 

                # Clear the adjoint solution as we need to recompute them 
                for i in range(adjointer.equation_count):
                    adjointer.forget_adjoint_values(i)

            self.set_local(m, m_array)
        self.H.update(m)

        m_dot = [copy_data(p.data()) for p in self.parameter] 
        self.set_local(m_dot, m_dot_array)

        hess = self.__base_hessian__(m_dot) 
        hess_array = get_global(hess)

        return hess_array
    
    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_parameters(self):
        m = [p.data() for p in self.parameter]
        return self.obj_to_array(m)

    def set_parameters(self, array):
        m = [p.data() for p in self.parameter]
        return self.set_local(m, array)

    def pyopt_problem(self, constraints=None, bounds=None, name="Problem", ignore_model_errors=False):
      '''Return a pyopt problem class that can be used with the PyOpt package,
      http://www.pyopt.org/
      '''
      import pyOpt
      import optimization.constraints

      constraints = optimization.constraints.canonicalise(constraints)

      def obj(x):
          ''' Evaluates the functional for the parameter choice x. '''

          fail = False
          if not ignore_model_errors:
              j = self(x)
          else:
              try:
                  j = self(x)
              except:
                  fail = True

          if constraints is not None:
              # Not sure how to do this in parallel, FIXME
              g = np.concatenate(constraints.function(x))
          else:
              g = [0]  # SNOPT fails if no constraints are given, hence add a dummy constraint

          return j, g, fail

      def grad(x, f, g):
          ''' Evaluates the gradient for the parameter choice x.
          f is the associated functional value and g are the values 
          of the constraints. '''

          fail = False
          if not ignore_model_errors:
              dj = self.derivative(x, forget=False)
          else:
              try:
                  dj = self.derivative(x, forget=False)
              except:
                  fail = True

          if constraints is not None:
              gJac = np.concatenate([gather(c.jacobian(x)) for c in constraints])
          else:
              gJac = np.zeros(len(x))  # SNOPT fails if no constraints are given, hence add a dummy constraint

          info("j = %f\t\t|dJ| = %f" % (f[0], np.linalg.norm(dj)))
          return np.array([dj]), gJac, fail


      # Instantiate the optimization problem
      opt_prob = pyOpt.Optimization(name, obj)
      opt_prob.addObj('J')

      # Compute bounds
      m = self.get_parameters() 
      n = len(m)

      if bounds is not None:
        bounds_arr = [None, None]  
        for i in range(2):
            if isinstance(bounds[i], float) or isinstance(bounds[i], int):
                bounds_arr[i] = np.ones(n) * bounds[i]
            else:
                bounds_arr[i] = np.array(bounds[i])
        lb, ub = bounds_arr 

      else:
        mx = np.finfo(np.double).max
        ub = mx * np.ones(n)

        mn = np.finfo(np.double).min
        lb = mn * np.ones(n)

      # Add parameters
      opt_prob.addVarGroup("variables", n, type='c', value=m, lower=lb, upper=ub)

      # Add constraints
      if constraints is not None:
          for i, c in enumerate(constraints):
              if isinstance(c, optimization.constraints.EqualityConstraint):
                opt_prob.addConGroup(str(i) + 'th constraint', c._get_constraint_dim(), type='e', equal=0.0)
              elif isinstance(c, optimization.constraints.InequalityConstraint):
                opt_prob.addConGroup(str(i) + 'th constraint', c._get_constraint_dim(), type='i', lower=0.0, upper=np.inf)

      return opt_prob, grad


def copy_data(m):
    ''' Returns a deep copy of the given Function/Constant. '''
    if hasattr(m, "vector"): 
        return Function(m.function_space())
    elif hasattr(m, "value_size"): 
        return Constant(m(()))
    elif hasattr(m, "copy"): 
        return m.copy()
    else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

def get_global(m_list):
    ''' Takes a list of distributed objects and returns one np array containing their (serialised) values '''
    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    m_global = []
    for m in m_list:

        # Parameters of type float
        if m == None or type(m) == float:
            m_global.append(m)

        elif hasattr(m, "tolist"): 
            m_global += m.tolist()

        # Function parameters of type Function 
        elif hasattr(m, "vector") or hasattr(m, "gather"): 
            if not hasattr(m, "gather"):
                m_v = m.vector()
            else:
                m_v = m
            m_a = gather(m_v)

            m_global += m_a.tolist()

        # Parameters of type Constant 
        elif hasattr(m, "value_size"): 
            a = np.zeros(m.value_size())
            p = np.zeros(m.value_size())
            m.eval(a, p)
            m_global += a.tolist()

        else:
            raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    return np.array(m_global, dtype='d')

def set_local(m_list, m_global_array):
    ''' Sets the local values of one or a list of distributed object(s) to the values contained in the global array m_global_array '''

    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    offset = 0
    for m in m_list:
        # Function parameters of type dolfin.Function 
        if hasattr(m, "vector"): 

            range_begin, range_end = m.vector().local_range()
            m_a_local = m_global_array[offset + range_begin:offset + range_end]
            m.vector().set_local(m_a_local)
            m.vector().apply('insert')
            offset += m.vector().size() 
        # Parameters of type dolfin.Constant 
        elif hasattr(m, "value_size"): 
            m.assign(constant.Constant(np.reshape(m_global_array[offset:offset+m.value_size()], m.shape())))
            offset += m.value_size()    
        elif isinstance(m, np.ndarray): 
            m[:] = m_global_array[offset:offset+len(m)]
            offset += len(m)
        else:
            raise TypeError, 'Unknown parameter type %s' % m.__class__


ReducedFunctionalNumpy = ReducedFunctionalNumPy
