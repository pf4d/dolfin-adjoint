import ufl
import ufl.classes
import ufl.algorithms
import ufl.operators

import dolfin.fem.solving
import dolfin

import libadjoint

import hashlib

debugging={}

class CoeffStore(object):
  '''This object manages the mapping from Dolfin coefficients to libadjoint Variables.
  In the process, it also manages the incrementing of the timestep associated with each
  variable, so that the user does not have to manually manage the time information.'''
  def __init__(self):
    self.coeffs={}

  def next(self, coeff):
    '''Increment the timestep corresponding to the provided Dolfin
    coefficient and then return the corresponding libadjoint variable.'''

    try:
      self.coeffs[str(coeff)]+=1
    except KeyError:
      self.coeffs[str(coeff)]=0

    return libadjoint.Variable(str(coeff), self.coeffs[str(coeff)], 0)

  def __getitem__(self, coeff):
    '''Return the libadjoint variable corresponding to coeff.'''

    if not self.coeffs.has_key(str(coeff)):
      self.coeffs[str(coeff)]=0

    return libadjoint.Variable(str(coeff), self.coeffs[str(coeff)], 0)

adj_variables=CoeffStore()

# Set record_all to true to enable recording all variables in the forward
# run. This is primarily useful for debugging.
debugging["record_all"] = False
debugging["test_hermitian"] = None

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()

def solve(*args, **kwargs):
  '''This solve routine comes from the dolfin_adjoint package, and wraps the real Dolfin
  solve call. Its purpose is to annotate the model, recording what solves occur and what
  forms are involved, so that the adjoint model may be constructed automatically by libadjoint. 

  To disable the annotation, just pass annotate=False to this routine, and it acts exactly
  like the Dolfin solve call. This is useful in cases where the solve is known to be irrelevant
  or diagnostic for the purposes of the adjoint computation (such as projecting fields to
  other function spaces for the purposes of visualisation).'''

  # First, decide if we should annotate or not.
  annotate = True
  if "annotate" in kwargs:
    annotate = kwargs["annotate"]
    del kwargs["annotate"] # so we don't pass it on to the real solver

  if annotate:
    if isinstance(args[0], ufl.classes.Equation):
      # annotate !

      # Unpack the arguments, using the same routine as the real Dolfin solve call
      unpacked_args = dolfin.fem.solving._extract_args(*args, **kwargs)
      eq = unpacked_args[0]
      u  = unpacked_args[1]
      bcs = unpacked_args[2]
      J = unpacked_args[3]

      # Set up the data associated with the matrix on the left-hand side. This goes on the diagonal
      # of the 'large' system that incorporates all of the timelevels, which is why it is prefixed
      # with diag.
      diag_name = hashlib.md5(str(eq.lhs)).hexdigest() # we don't have a useful human-readable name, so take the md5sum of the string representation of the form
      diag_deps = [adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(eq.lhs) if hasattr(coeff, "function_space")]
      diag_coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(eq.lhs) if hasattr(coeff, "function_space")]
      diag_block = libadjoint.Block(diag_name, dependencies=diag_deps, test_hermitian=debugging["test_hermitian"])

      # Similarly, create the object associated with the right-hand side data.
      rhs=RHS(eq.rhs)

      # We need to check if this is the first equation,
      # so that we can register the appropriate initial conditions.
      # These equations are necessary so that libadjoint can assemble the
      # relevant adjoint equations for the adjoint variables associated with
      # the initial conditions.
      for coeff, dep in zip(rhs.coefficients(),rhs.dependencies()) + zip(diag_coeffs, diag_deps):
        # If coeff is not known, it must be an initial condition.
        if not adjointer.variable_known(dep):
          fn_space = coeff.function_space()
          block_name = "Identity: %s" % str(fn_space)
          identity_block = libadjoint.Block(block_name)
        
          init_rhs=Vector(coeff).duplicate()
          init_rhs.axpy(1.0,Vector(coeff))

          def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
            assert coefficient == 1
            return (Matrix(ufl.Identity(fn_space.dim())), Vector(dolfin.Function(fn_space)))

          identity_block.assemble=identity_assembly_cb

          if debugging["record_all"]:
            adjointer.record_variable(dep, libadjoint.MemoryStorage(Vector(coeff)))

          initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=RHS(init_rhs))
          adjointer.register_equation(initial_eq)
          assert adjointer.variable_known(dep)

      var = adj_variables.next(u)

      # With the initial conditions out of the way, let us now define the callbacks that
      # define the action of the operator the user has passed in on the lhs of this equation.

      def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
        '''This callback must conform to the libadjoint Python block assembly
        interface. It returns either the form or its transpose, depending on
        the value of the logical hermitian.'''

        assert coefficient == 1

        value_coeffs=[v.data for v in values]

        eq_l=dolfin.replace(eq.lhs, dict(zip(diag_coeffs, value_coeffs)))

        if hermitian:
          # Homogenise the adjoint boundary conditions. This creates the adjoint
          # solution associated with the lifted discrete system that is actually solved.
          adjoint_bcs = [dolfin.homogenize(bc) for bc in bcs if isinstance(bc, dolfin.DirichletBC)]
          if len(adjoint_bcs) == 0: adjoint_bcs = None
          return (Matrix(dolfin.adjoint(eq_l), bcs=adjoint_bcs), Vector(None, fn_space=u.function_space()))
        else:
          return (Matrix(eq_l, bcs=bcs), Vector(None, fn_space=u.function_space()))
      diag_block.assemble=diag_assembly_cb

      if len(diag_deps) > 0:
        def derivative_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
          dolfin_variable = values[dependencies.index(variable)].data
          dolfin_values = [val.data for val in values]

          current_form = dolfin.replace(eq.lhs, dict(zip(diag_coeffs, dolfin_values)))

          deriv = ufl.derivative(current_form, dolfin_variable, contraction_vector.data)

          if hermitian:
            deriv = dolfin.adjoint(deriv, reordered_arguments=ufl.algorithms.extract_arguments(deriv))

          action = coefficient * dolfin.action(deriv, input.data)

          return Vector(action)
        diag_block.derivative_action = derivative_action

      eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)

      adjointer.register_equation(eqn)

  dolfin.fem.solving.solve(*args, **kwargs)

  # Finally, if we want to record all of the solutions of the real forward model
  # (for comparison with a libadjoint replay later),
  # then we should record the value of the variable we just solved for.
  if isinstance(args[0], ufl.classes.Equation) and annotate:
    if debugging["record_all"]:
      adjointer.record_variable(var, libadjoint.MemoryStorage(Vector(u)))


def adj_html(*args, **kwargs):
  '''This routine dumps the current state of the adjointer to a HTML visualisation.
  Use it like:
    adj_html("forward.html", "forward") # for the equations recorded on the forward run
    adj_html("adjoint.html", "adjoint") # for the equations to be assembled on the adjoint run
  '''
  return adjointer.to_html(*args, **kwargs)


class Vector(libadjoint.Vector):
  '''This class implements the libadjoint.Vector abstract base class for the Dolfin adjoint.
  In particular, it must implement the data callbacks for tasks such as adding two vectors
  together, duplicating vectors, taking norms, etc., that occur in the process of constructing
  the adjoint equations.'''

  def __init__(self, data, zero=False, fn_space=None):

    self.data=data
    # self.zero is true if we can prove that the vector is zero.
    if data is None:
      self.zero=True
    else:
      self.zero=zero

    if fn_space is not None:
      self.fn_space = fn_space

  def duplicate(self):

    if isinstance(self.data, ufl.form.Form):
      # The data type will be determined by the first addto.
      data = None
    elif isinstance(self.data, dolfin.Function):
      data = dolfin.Function(self.data.function_space())
    else:
      data = None

    fn_space = None
    if hasattr(self, "fn_space"):
      fn_space = self.fn_space

    return Vector(data, zero=True, fn_space=fn_space)

  def axpy(self, alpha, x):

    if x.zero:
      return

    if (self.data is None):
      # self is an empty form.
      assert(isinstance(x.data, ufl.form.Form))
      self.data=alpha*x.data
    elif isinstance(self.data, dolfin.Coefficient):
      if isinstance(x.data, dolfin.Coefficient):
        self.data.vector().axpy(alpha, x.data.vector())
      else:
        # This occurs when adding a RHS derivative to an adjoint equation
        # corresponding to the initial conditions.
        self.data.vector().axpy(alpha, dolfin.assemble(x.data))
    else:
      # self is a non-empty form.
      assert(isinstance(x.data, ufl.form.Form))
      assert(isinstance(self.data, ufl.form.Form))
      self.data+=alpha*x.data

    self.zero = False

  def norm(self):

    return (abs(dolfin.assemble(dolfin.inner(self.data, self.data)*dolfin.dx)))**0.5

  def dot_product(self,y):

    if isinstance(self.data, ufl.form.Form):
      return dolfin.assemble(dolfin.inner(self.data, y.data)*dolfin.dx)
    elif isinstance(self.data, dolfin.Function):
      import numpy
      if isinstance(y.data, ufl.form.Form):
        other = dolfin.assemble(y.data)
      else:
        other = y.data.vector()
      return numpy.dot(numpy.array(self.data.vector()), numpy.array(other))
    else:
      raise LibadjointErrorNotImplemented("Don't know how to dot anything else.")

  def set_random(self):
    assert isinstance(self.data, dolfin.Function) or hasattr(self, "fn_space")
    import random

    if self.data is None:
      self.data = dolfin.Function(self.fn_space)

    vec = self.data.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

class Matrix(libadjoint.Matrix):
  '''This class implements the libadjoint.Matrix abstract base class for the Dolfin adjoint.
  In particular, it must implement the data callbacks for tasks such as adding two matrices
  together, duplicating matrices, etc., that occur in the process of constructing
  the adjoint equations.'''

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
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint.
  It takes in a form, and implements the necessary routines such as calling the functional
  and taking its derivative.'''

  def __init__(self, form):

    self.form=form

  def __call__(self, dependencies, values):

    dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

    dolfin_values=[val.data for val in values]

    return dolfin.assemble(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

    dolfin_values = [val.data for val in values]

    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
    test = dolfin.TestFunction(dolfin_variable.function_space())

    return Vector(ufl.derivative(current_form, dolfin_variable, test))

  def dependencies(self, adjointer, timestep):

    if timestep == adjointer.timestep_count-1:
      deps = [adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
    else:
      deps = []
    
    return deps

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()

class RHS(libadjoint.RHS):
  '''This class implements the libadjoint.RHS abstract base class for the Dolfin adjoint.
  It takes in a form, and implements the necessary routines such as calling the right-hand side
  and taking its derivative.'''
  def __init__(self, form):

    self.form=form

    if isinstance(self.form, ufl.form.Form):
      self.deps = [adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
    else:
      self.deps = []
    
    if isinstance(self.form, ufl.form.Form):
      self.coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
    else:
      self.coeffs = []

  def __call__(self, dependencies, values):

    if isinstance(self.form, ufl.form.Form):

      dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]
      
      dolfin_values=[val.data for val in values]

      return Vector(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

    else:
      # RHS is a Vector.
      assert isinstance(self.form, Vector)
      return self.form
    

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

    if isinstance(self.form, ufl.form.Form):
      # Find the dolfin Function corresponding to variable.
      dolfin_variable = values[dependencies.index(variable)].data

      dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

      dolfin_values = [val.data for val in values]

      current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
      trial = dolfin.TrialFunction(dolfin_variable.function_space())

      d_rhs=ufl.derivative(current_form, dolfin_variable, trial)

      if hermitian:
        action = dolfin.action(dolfin.adjoint(d_rhs, reordered_arguments=ufl.algorithms.extract_arguments(d_rhs)),contraction_vector.data)
      else:
        action = dolfin.action(d_rhs,contraction_vector.data)

      return Vector(action)
    else:
      # RHS is a Vector. Its derivative is therefore zero.
      raise exceptions.LibadjointErrorNotImplemented("No derivative method for constant RHS.")


  def dependencies(self):

    return self.deps

  def coefficients(self):

    return self.coeffs

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()

dolfin_assign = dolfin.Function.assign
def dolfin_adjoint_assign(self, other, annotate=True):
  '''We also need to monkeypatch the Function.assign method, as it is often used inside 
  the main time loop, and not annotating it means you get the adjoint wrong for totally
  nonobvious reasons. If anyone objects to me monkeypatching your objects, my apologies
  in advance.'''

  # ignore anything not a dolfin.Function
  if not isinstance(other, dolfin.Function) or annotate is False:
    return dolfin_assign(self, other)

  # ignore anything that is an interpolation, rather than a straight assignment
  if self.function_space() != other.function_space():
    return dolfin_assign(self, other)

  other_var = adj_variables[other]
  # ignore any functions we haven't seen before -- we DON'T want to
  # annotate the assignment of initial conditions here. That happens
  # in the main solve wrapper.
  if not adjointer.variable_known(other_var):
    return dolfin_assign(self, other)

  # OK, so we have a variable we've seen before. Beautiful.
  fn_space = other.function_space()
  u, v = dolfin.TestFunction(fn_space), dolfin.TrialFunction(fn_space)
  M = dolfin.inner(u, v)*dolfin.dx
  return solve(M == dolfin.action(M, other), self) # this takes care of all the annotation etc

  # -----------------------------------------------------------
  #block_name = "Identity: %s" % str(fn_space)

  #diag_identity_block = libadjoint.Block(block_name)
  #def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
  #  assert coefficient == 1
  #  return (Matrix(ufl.Identity(fn_space.dim())), Vector(dolfin.Function(fn_space)))
  #diag_identity_block.assemble = identity_assembly_cb

  #offdiag_identity_block = libadjoint.Block(block_name, coefficient=-1.0)
  #def identity_action_cb(variables, dependencies, hermitian, coefficient, input, context):
  #  new = input.duplicate()
  #  new.axpy(coefficient, input)
  #  return new
  #offdiag_identity_block.action = identity_action_cb

  #self_var = adj_variables.next(self)

  #if debugging["record_all"]:
  #  adjointer.record_variable(self_var, libadjoint.MemoryStorage(Vector(other)))

  #assign_eq = libadjoint.Equation(self_var, blocks=[offdiag_identity_block, diag_identity_block], targets=[other_var, self_var])
  #adjointer.register_equation(assign_eq)
  # -----------------------------------------------------------

  # And we're done.

  # return dolfin_assign(self, other)

dolfin.Function.assign = dolfin_adjoint_assign
