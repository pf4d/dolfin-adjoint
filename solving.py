import ufl
import ufl.classes
import ufl.algorithms
import ufl.operators

import dolfin.fem.solving
import dolfin

import libadjoint

import hashlib

debugging={}

# Set record_all to true to enable recording all variables in the forward
# run. This is primarily useful for debugging.
debugging["record_all"] = False

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
        return (Matrix(dolfin.fem.formmanipulations.adjoint(eq_l)), Vector(None))
      else:
        return (Matrix(eq_l, bcs=bcs), Vector(None))

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

        phi=dolfin.TestFunction(fn_space)
        psi=dolfin.TrialFunction(fn_space)

        rhs=Vector(rhs_coeffs[index]).duplicate()
        rhs.axpy(1.0,Vector(rhs_coeffs[index]))

        def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):

          assert coefficient == 1
          #return (Matrix(dolfin.inner(phi,psi)*dolfin.dx), Vector(None))
          return (Matrix(ufl.Identity(fn_space.dim())), Vector(dolfin.Function(fn_space)))
        
        identity_block.assemble=identity_assembly_cb

        def init_rhs_cb(adjointer, variable, dependencies, values, context):
          return rhs

        if debugging["record_all"]:
          adjointer.record_variable(libadjoint.Variable(rhs_coeffs[index].adj_name,rhs_coeffs[index].adj_timestep), 
                                    libadjoint.MemoryStorage(Vector(rhs_coeffs[index])))

        initial_eq = libadjoint.Equation(rhs_dep, blocks=[identity_block], targets=[rhs_dep], rhs_cb=init_rhs_cb)
        adjointer.register_equation(initial_eq)

    adjointer.register_equation(eqn)

  dolfin.fem.solving.solve(*args, **kwargs)

  if isinstance(args[0], ufl.classes.Equation):
    
    if debugging["record_all"]:
      adjointer.record_variable(libadjoint.Variable(u.adj_name,u.adj_timestep), libadjoint.MemoryStorage(Vector(u)))

def adj_variable_from_coeff(coeff):
  try:
    iteration = coeff.adj_iteration
  except AttributeError:
    iteration = 0

  return libadjoint.Variable(coeff.adj_name, coeff.adj_timestep, iteration)

def adj_html(*args, **kwargs):
  return adjointer.to_html(*args, **kwargs)


class Vector(libadjoint.Vector):
  def __init__(self, data, zero=False):

    self.data=data
    # self.zero is true if we can prove that the vector is zero.
    if data is None:
      self.zero=True
    else:
      self.zero=zero

  def duplicate(self):
    '''The data type will be determined by the first addto.'''

    if isinstance(self.data, ufl.form.Form):
      data=None
    else:
      data=dolfin.Function(self.data.function_space())

    return Vector(data, zero=True)

  def axpy(self, alpha, x):

    if x.zero:
      return

    if (self.data is None):
      # self is an empty form.
      assert(isinstance(x.data, ufl.form.Form))
      self.data=alpha*x.data
    elif isinstance(self.data, dolfin.Coefficient):
      self.data.vector().axpy(alpha, x.data.vector())
    else:
      # self is a non-empty form.
      assert(isinstance(x.data, ufl.form.Form))
      assert(isinstance(self.data, ufl.form.Form))
      self.data+=alpha*x.data

    self.zero = False

  def norm(self):

    return (abs(dolfin.assemble(dolfin.inner(self.data, self.data)*dolfin.dx)))**0.5

class Matrix(libadjoint.Matrix):
  def __init__(self, data, bcs=None):

    self.bcs=bcs

    self.data=data

  def solve(self, b):
      
    if isinstance(self.data, ufl.Identity):
      x=b.duplicate()
      x.axpy(1.0, b)
    else:
      x=Vector(dolfin.Function(self.test_function().function_space()))
      dolfin.fem.solving.solve(self.data==b.data, x.data, self.bcs)

    return x

  def test_function(self):
    '''test_function(self)

    Return the ufl.Argument corresponding to the trial space for the form'''

    return ufl.algorithms.extract_arguments(self.data)[-1]

class Functional(libadjoint.Functional):
  def __init__(self, form):

    self.form=form

  def __call__(self, dependencies, values):

    dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "adj_timestep")]

    dolfin_values=[val.data for val in values]

    return dolfin.assemble(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "adj_timestep")]

    dolfin_values = [val.data for val in values]

    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
    test = dolfin.TestFunction(dolfin_variable.function_space())

    return Vector(ufl.derivative(current_form, dolfin_variable, test))

  def dependencies(self, adjointer, timestep):

    if timestep == adjointer.timestep_count-1:
      deps = [adj_variable_from_coeff(coeff) for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "adj_timestep")]      
    else:
      deps = []
    
    return deps

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()
