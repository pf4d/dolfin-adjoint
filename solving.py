import ufl
import ufl.classes
import ufl.algorithms
import ufl.operators

import dolfin.fem.solving
import dolfin

import libadjoint

import hashlib

adjointer = libadjoint.Adjointer()

def solve(*args, **kwargs):
  if isinstance(args[0], ufl.classes.Equation):
    # annotate !
    unpacked_args = dolfin.fem.solving._extract_args(*args, **kwargs)
    eq = unpacked_args[0]
    u  = unpacked_args[1]
    bcs = unpacked_args[2]
    J = unpacked_args[3]

    diag_name = hashlib.md5(str(eq.lhs)).hexdigest()
    diag_deps = [adj_variable_from_coeff(coeff) for coeff in ufl.algorithms.extract_coefficients(eq.lhs) if hasattr(coeff, "adj_timestep")]
    diag_coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(eq.lhs) if hasattr(coeff, "adj_timestep")]
    diag_block = libadjoint.Block(diag_name, dependencies=diag_deps)

    var = adj_variable_from_coeff(u)

    rhs_deps = [adj_variable_from_coeff(coeff) for coeff in ufl.algorithms.extract_coefficients(eq.rhs) if hasattr(coeff, "adj_timestep")]
    rhs_coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(eq.rhs) if hasattr(coeff, "adj_timestep")]

    def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):

      assert coefficient == 1
      fn_space = u.function_space()

      value_coeffs=[v.data for v in values]

      eq_l=dolfin.replace(eq.lhs, dict(zip(diag_coeffs, value_coeffs)))

      if hermitian:
        return (Matrix(ufl.operators.transpose(eq_l)), Vector(dolfin.Function(fn_space)))
      else:
        return (Matrix(eq_l, bcs=bcs), Vector(dolfin.Function(fn_space)))

    def rhs_cb(adjointer, variable, dependencies, values, context):
      # 
      value_coeffs=[v.data for v in values]

      return Vector(dolfin.replace(eq.rhs, dict(zip(rhs_coeffs, value_coeffs))))

    diag_block.assemble=diag_assembly_cb

    eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs_deps=rhs_deps, rhs_cb=rhs_cb)

    # we need to check if this is the first equation,
    # so that we can register the appropriate initial conditions
    if adjointer.equation_count == 0:
      for index, rhs_dep in enumerate(rhs_deps):
        assert rhs_dep.timestep == 0
        fn_space = rhs_coeffs[index].function_space()
        block_name = "Identity: %s" % str(fn_space)
        identity_block = libadjoint.Block(block_name)

        def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):

          assert coefficient == 1
          return (Matrix(ufl.Identity(fn_space.dim())), Vector(dolfin.Function(fn_space)))
        
        identity_block.assemble=identity_assembly_cb

        def zero_rhs_cb(adjointer, variable, dependencies, values, context):
          return None

        initial_eq = libadjoint.Equation(rhs_dep, blocks=[identity_block], targets=[rhs_dep], rhs_cb=zero_rhs_cb)
        adjointer.register_equation(initial_eq)

    adjointer.register_equation(eqn)

  dolfin.fem.solving.solve(*args, **kwargs)

def adj_variable_from_coeff(coeff):
  try:
    iteration = coeff.adj_iteration
  except AttributeError:
    iteration = 0

  return libadjoint.Variable(coeff.adj_name, coeff.adj_timestep, iteration)

def adj_html(*args, **kwargs):
  return adjointer.to_html(*args, **kwargs)


class Vector(libadjoint.Vector):
  def __init__(self, data):

    self.data=data

  def duplicate(self):
    
    data=dolfin.Function(self.data.function_space())

    return Vector(data)

  def axpy(self, alpha, x):
  
    self.data+=alpha*x.data

class Matrix(libadjoint.Matrix):
  def __init__(self, data, bcs=None):

    self.bcs=bcs

    self.data=data

  def solve(self, b):
      
    x=b.duplicate()

    if isinstance(self.data, ufl.Identity):
      x.data.assign(b.data)
    else:
      dolfin.fem.solving.solve(self.data==b.data, x.data, self.bcs)

    return x
