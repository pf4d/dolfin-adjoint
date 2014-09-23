import backend
import utils
import adjglobals
import solving
import libadjoint
import adjlinalg

if hasattr(backend, 'FunctionAssigner'):
  class FunctionAssigner(backend.FunctionAssigner):

    def __init__(self, *args, **kwargs):
        super(FunctionAssigner, self).__init__(*args, **kwargs)

        # The adjoint function assigner with swapped FunctionSpace arguments
        self.adj_function_assigner = backend.FunctionAssigner(args[1], args[0])

    def assign(self, receiving, giving, annotate=None):

      out = backend.FunctionAssigner.assign(self, receiving, giving)
      to_annotate = utils.to_annotate(annotate)

      if to_annotate:
        # Receiving is always a single Function, or a single Function.sub(foo).sub(bar)

        # If the user passes in v.sub(0) to this function, we need to get a handle on v;
        # fetch that now
        receiving_super = get_super_function(receiving)
        receiving_fnspace = receiving_super.function_space()
        receiving_identity = utils.get_identity_block(receiving_fnspace)
        receiving_idx = get_super_idx(receiving)

        rhs = FunctionAssignerRHS(self, self.adj_function_assigner, receiving_super,
                receiving_idx, giving)
        receiving_dep = adjglobals.adj_variables.next(receiving_super)

        solving.register_initial_conditions(zip(rhs.coefficients(), rhs.dependencies()), linear=True)
        if backend.parameters["adjoint"]["record_all"]:
          adjglobals.adjointer.record_variable(receiving_dep, libadjoint.MemoryStorage(adjlinalg.Vector(receiving_super)))

        eq = libadjoint.Equation(receiving_dep, blocks=[receiving_identity], targets=[receiving_dep], rhs=rhs)
        cs = adjglobals.adjointer.register_equation(eq)

        solving.do_checkpoint(cs, receiving_dep, rhs)

      return out

  class FunctionAssignerRHS(libadjoint.RHS):
    def __init__(self, function_assigner, adj_function_assigner, receiving_super, receiving_idx, giving):
      self.function_assigner = function_assigner
      self.adj_function_assigner = adj_function_assigner

      self.receiving_super = receiving_super
      self.receiving_idx   = receiving_idx
      self.receiving_dep   = adjglobals.adj_variables[receiving_super]

      self.giving_supers   = get_super_function(giving)
      self.giving_idxs     = get_super_idx(giving)
      self.giving_deps     = adjglobals.adj_variables[self.giving_supers]


    def __call__(self, dependencies, values, hermitian=False):

      if not hermitian:
        receiving_idx = self.receiving_idx
        giving_idxs = self.giving_idxs
        rec_idx = 1
        giv_idx = 0
      else:
        receiving_idx = self.giving_idxs
        giving_idxs = self.receiving_idx
        rec_idx = 0
        giv_idx = 1

      receiving_super = backend.Function(values[rec_idx].data) # make a copy of the OLD value of what we're assigning to
      receiving_sub = receiving_super

      for idx in receiving_idx:
        receiving_sub = receiving_sub.sub(idx)

      giving = values[giv_idx].data

      for idx in giving_idxs:
          giving = giving.sub(idx)
      giving_subs = giving

      if not hermitian:
        self.function_assigner.assign(receiving_sub, giving_subs)
        print "not in adjoint. assigner returns", receiving_super.vector().array()

      else:
        self.adj_function_assigner.assign(receiving_sub, giving_subs)
        print "in adjoint. assigner returns", receiving_super.vector().array()

      return adjlinalg.Vector(receiving_super)

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
        # FunctionAssigner.assign is linear, which makes the tangent linearisation equivalent to
        # just calling it with the right args.


        new_values = []

        for (dep, value) in zip(dependencies, values):
          if dep == variable:
            new_values.append(contraction_vector)
            print "in derivative action. length of derivative variable vector:", len(value.data.vector())
            print "in derivative action. length of contraction vector:", len(contraction_vector.data.vector())
            assert len(value.data.vector()) == len(contraction_vector.data.vector())
          else:
            new_values.append(value.duplicate())

        return self.__call__(dependencies, new_values, hermitian)

    def dependencies(self):
      # This depends on the OLD value of the receiving function -- why?
      # Because: if we only update v.sub(0), v_new's other components
      # 1, 2, 3, ... are taken from v_old.
      return [self.giving_deps] + [self.receiving_dep]

    def coefficients(self):
      return [self.giving_supers] + [self.receiving_super]

    def __str__(self):
      return "FunctionAssignerRHS"

  def get_super_function(receiving):
    out = receiving
    while hasattr(out, 'super_fn'):
      out = out.super_fn
    return out

  def get_super_idx(receiving):
    indices = []
    sup = receiving
    while hasattr(sup, 'super_fn'):
      indices.append(sup.super_idx)
      sup = sup.super_fn

    return list(reversed(indices))
