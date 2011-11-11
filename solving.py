import ufl
import ufl.classes
import ufl.algorithms

import dolfin.fem.solving

import libadjoint

import hashlib

adjointer = libadjoint.Adjointer()

def solve(*args, **kwargs):
  if isinstance(args[0], ufl.classes.Equation):
    # annotate !
    eq, u, bcs, J, tol, M = dolfin.fem.solving._extract_args(*args, **kwargs)

    diag_name = hashlib.md5(str(eq.lhs)).hexdigest()
    diag_deps = [adj_variable_from_coeff(coeff) for coeff in ufl.algorithms.extract_coefficients(eq.lhs) if hasattr(coeff, "adj_timestep")]
    diag_block = libadjoint.Block(diag_name, dependencies=diag_deps)

    var = adj_variable_from_coeff(u)

    rhs_deps = [adj_variable_from_coeff(coeff) for coeff in ufl.algorithms.extract_coefficients(eq.rhs) if hasattr(coeff, "adj_timestep")]
    eq = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs_deps=rhs_deps)

    # we need to check if this is the first equation,
    # so that we can register the appropriate initial conditions
    if adjointer.equation_count == 0:
      for rhs_dep in rhs_deps:
        assert rhs_dep.timestep == 0
        identity_block = libadjoint.Block("Identity")
        initial_eq = libadjoint.Equation(rhs_dep, blocks=[identity_block], targets=[rhs_dep])
        adjointer.register_equation(initial_eq)

    adjointer.register_equation(eq)

  dolfin.fem.solving.solve(*args, **kwargs)

def adj_variable_from_coeff(coeff):
  try:
    iteration = coeff.adj_iteration
  except AttributeError:
    iteration = 0

  return libadjoint.Variable(coeff.adj_name, coeff.adj_timestep, iteration)

def adj_html(*args, **kwargs):
  return adjointer.to_html(*args, **kwargs)
