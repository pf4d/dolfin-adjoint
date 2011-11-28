import ufl
import ufl.classes
import ufl.algorithms

import dolfin.fem.solving
import dolfin

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
    rhs_coeffs = ufl.algorithms.extract_coefficients(eq.rhs)

    def rhs_cb(adjointer, variable, dependencies, values, context):
      # Need to replace the arguments (implicitly stored in eq.rhs) with the values from the values array!
      # Otherwise it will evaluate things wrongly, I think ...
      return eq.rhs

    eq = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs_deps=rhs_deps, rhs_cb=rhs_cb)

    # we need to check if this is the first equation,
    # so that we can register the appropriate initial conditions
    if adjointer.equation_count == 0:
      for index, rhs_dep in enumerate(rhs_deps):
        assert rhs_dep.timestep == 0
        fn_space = rhs_coeffs[index].function_space()
        block_name = "Identity: %s" % str(fn_space)
        identity_block = libadjoint.Block(block_name)

        def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
          print "Inside identity_assembly_cb"
          assert coefficient == 1
          return (ufl.Identity(fn_space.dim()), dolfin.Function(fn_space))

        adjointer.register_block_assembly_callback(block_name, identity_assembly_cb)

        def zero_rhs_cb(adjointer, variable, dependencies, values, context):
          return None

        initial_eq = libadjoint.Equation(rhs_dep, blocks=[identity_block], targets=[rhs_dep], rhs_cb=zero_rhs_cb)
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
