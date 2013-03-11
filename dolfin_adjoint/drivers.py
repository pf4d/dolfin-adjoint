import libadjoint
from parameter import *
from dolfin import info_red, info_blue, info
import adjglobals
import dolfin
import numpy
import constant
import adjresidual
import ufl.algorithms

def replay_dolfin(forget=False, tol=0.0, stop=False):
  if not dolfin.parameters["adjoint"]["record_all"]:
    info_red("Warning: your replay test will be much more effective with dolfin.parameters['adjoint']['record_all'] = True.")

  success = True
  for i in range(adjglobals.adjointer.equation_count):
      (fwd_var, output) = adjglobals.adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=tol)
      storage.set_overwrite(True)
      out = adjglobals.adjointer.record_variable(fwd_var, storage)
      success = success and out

      if forget:
        adjglobals.adjointer.forget_forward_equation(i)

      if not out and stop:
        return success

  return success

def compute_adjoint(functional, forget=True, ignore=[]):

  ignorelist = []
  for fn in ignore:
    if isinstance(fn, dolfin.Function):
      ignorelist.append(adjglobals.adj_variables[fn])
    elif isinstance(fn, str):
      ignorelist.append(libadjoint.Variable(fn, 0, 0))
    else:
      ignorelist.append(fn)

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(functional, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
      fwd_var = adjglobals.adjointer.get_forward_variable(i)
      if fwd_var in ignorelist:
        info("Ignoring the adjoint equation for %s" % fwd_var)
        continue

      (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, functional)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(adj_var, storage)

      # forget is None: forget *nothing*.
      # forget is True: forget everything we can, forward and adjoint
      # forget is False: forget only unnecessary adjoint values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_adjoint_equation(i)
      else:
        adjglobals.adjointer.forget_adjoint_values(i)

      yield (output.data, adj_var)

def compute_tlm(parameter, forget=False):

  for i in range(adjglobals.adjointer.equation_count):
      (tlm_var, output) = adjglobals.adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(tlm_var, storage)

      # forget is None: forget *nothing*.
      # forget is True: forget everything we can, forward and adjoint
      # forget is False: forget only unnecessary tlm values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_tlm_equation(i)
      else:
        adjglobals.adjointer.forget_tlm_values(i)

      yield (output.data, tlm_var)

def compute_gradient(J, param, forget=True, ignore=[], callback=lambda var, output: None):
  dolfin.parameters["adjoint"]["stop_annotating"] = True

  try:
    scalar = False
    dJdparam = [None for i in range(len(param))]
    lparam = param
  except TypeError:
    scalar = True
    dJdparam = [None]
    lparam = [param]
  last_timestep = adjglobals.adjointer.timestep_count

  ignorelist = []
  for fn in ignore:
    if isinstance(fn, dolfin.Function):
      ignorelist.append(adjglobals.adj_variables[fn])
    elif isinstance(fn, str):
      ignorelist.append(libadjoint.Variable(fn, 0, 0))
    else:
      ignorelist.append(fn)

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(J, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
    fwd_var = adjglobals.adjointer.get_forward_variable(i)
    if fwd_var in ignorelist:
      info("Ignoring the adjoint equation for %s" % fwd_var)
      continue

    (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, J)

    callback(adj_var, output.data)

    storage = libadjoint.MemoryStorage(output)
    storage.set_overwrite(True)
    adjglobals.adjointer.record_variable(adj_var, storage)
    fwd_var = libadjoint.Variable(adj_var.name, adj_var.timestep, adj_var.iteration)

    for j in range(len(lparam)):
      out = lparam[j].equation_partial_derivative(adjglobals.adjointer, output.data, i, fwd_var)
      dJdparam[j] = _add(dJdparam[j], out)

      if last_timestep > adj_var.timestep:
        # We have hit a new timestep, and need to compute this timesteps' \partial J/\partial m contribution
        last_timestep = adj_var.timestep
        out = lparam[j].functional_partial_derivative(adjglobals.adjointer, J, adj_var.timestep)
        dJdparam[j] = _add(dJdparam[j], out)

    if forget is None:
      pass
    elif forget:
      adjglobals.adjointer.forget_adjoint_equation(i)
    else:
      adjglobals.adjointer.forget_adjoint_values(i)

  if scalar:
    return dJdparam[0]
  else:
    return dJdparam

def hessian(J, m, policy="default", warn=True):
  '''Choose which Hessian the user wants.'''
  dolfin.parameters["adjoint"]["stop_annotating"] = True
  return BasicHessian(J, m, warn=warn)

class BasicHessian(libadjoint.Matrix):
  '''A basic implementation of the Hessian class that recomputes the tangent linear, adjoint and second-order adjoint
  equations on each action. Should be the slowest, but safest, with the lowest memory requirements.'''
  def __init__(self, J, m, warn=True):
    self.J = J
    self.m = m

    if warn:
      dolfin.info_red("Warning: Hessian computation is still experimental and is known to not work for some problems. Please Taylor test thoroughly.")

    if not isinstance(m, (InitialConditionParameter, ScalarParameter)):
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, Hessian computation only works for InitialConditionParameter|SteadyParameter|TimeConstantParameter|ScalarParameter so far.")

    self.update(m)

  def update(self, m):
    pass

  def __call__(self, m_dot):

    m_p = self.m.set_perturbation(m_dot)
    last_timestep = adjglobals.adjointer.timestep_count

    if hasattr(m_dot, 'function_space'):
      Hm = dolfin.Function(m_dot.function_space())
    elif isinstance(m_dot, float):
      Hm = 0.0
    else:
      raise NotImplementedError("Sorry, don't know how to handle this")

    # run the tangent linear model
    for (tlm, tlm_var) in compute_tlm(m_p, forget=None):
      pass

    # run the adjoint and second-order adjoint equations.
    i = adjglobals.adjointer.equation_count
    for (adj, adj_var) in compute_adjoint(self.J, forget=None):
      i = i - 1
      (soa_var, soa_vec) = adjglobals.adjointer.get_soa_solution(i, self.J, m_p)
      soa = soa_vec.data

      # now implement the Hessian action formula.
      out = self.m.equation_partial_derivative(adjglobals.adjointer, soa, i, soa_var.to_forward())
      if out is not None:
        if isinstance(Hm, dolfin.Function):
          Hm.vector().axpy(1.0, out.vector())
        elif isinstance(Hm, float):
          Hm += out

      out = self.m.equation_partial_second_derivative(adjglobals.adjointer, adj, i, soa_var.to_forward(), m_dot)
      if out is not None:
        if isinstance(Hm, dolfin.Function):
          Hm.vector().axpy(1.0, out.vector())
        elif isinstance(Hm, float):
          Hm += out

      if last_timestep > adj_var.timestep:
        # We have hit a new timestep, and need to compute this timesteps' \partial^2 J/\partial m^2 contribution
        last_timestep = adj_var.timestep
        out = self.m.functional_partial_second_derivative(adjglobals.adjointer, self.J, adj_var.timestep, m_dot)
        if out is not None:
          if isinstance(Hm, dolfin.Function):
            Hm.vector().axpy(1.0, out.vector())
          elif isinstance(Hm, float):
            Hm += out

      storage = libadjoint.MemoryStorage(soa_vec)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(soa_var, storage)

    return Hm

  def action(self, x, y):
    assert isinstance(x.data, dolfin.Function)
    assert isinstance(y.data, dolfin.Function)

    Hm = self.__call__(x.data)
    y.data.assign(Hm)

  def eigendecomposition(self, **kwargs):
    '''Compute the eigendecomposition of the Hessian.'''

    params = {'solver': 'krylovschur',
              'spectrum': 'largest magnitude',
              'type': 'hermitian',
              'monitor': True,
              'n': 1}

    params.update(kwargs)

    # We take in the options in "DOLFIN" syntax (the same
    # as SLEPcEigenSolver. libadjoint uses a different
    # syntax for the same things. Here we translate.
    # Sorry for the confusion.

    pairs = {'method': 'solver',
             'type': 'type',
             'which': 'spectrum',
             'monitor': 'monitor',
             'neigenpairs': 'n'}

    options = {}
    for key in pairs:
      options[key] = params[pairs[key]]

    # OK! Now add the model input and output vectors.
    data = adjlinalg.Vector(self.m.data())
    options['input'] = data
    options['output'] = data

    eps = adjglobals.adjointer.compute_eps(self, options)

    retval = []
    for i in range(eps.ncv):
      (lamda, u) = eps.get_eps(i)
      retval += [(lamda, u.data)]

    return retval

def _add(value, increment):
  # Add increment to value correctly taking into account None.
  if increment is None:
    return value
  elif value is None:
    return increment
  else:
    return value+increment

