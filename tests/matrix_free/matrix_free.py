from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True

mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
u = TrialFunction(V); v = TestFunction(V);

def main(ic):
  A = assemble(inner(grad(u), grad(v))*dx)
  bc.apply(A)
  b = assemble(ic * v* dx)
  bc.apply(b)

  class KrylovMatrix(PETScKrylovMatrix) :
      def __init__(self) :
          PETScKrylovMatrix.__init__(self, V.dim(), V.dim())

      def mult(self, *args):
          y = PETScVector(V.dim())
          A.mult(args[0], y)
          args[1].set_local(y.array())

      def transpmult(self, *args):
          y = PETScVector(V.dim())
          A.transpmult(args[0], y)
          args[1].set_local(y.array())

      def dependencies(self):
        return []

  y = Function(V)
  solve(A, y.vector(), b, "cg", "none")

  x = Function(V)
  KrylovSolver = AdjointPETScKrylovSolver("cg","none")
  KrylovSolver.solve(KrylovMatrix(), down_cast(x.vector()), down_cast(b))

  assert (x.vector() - y.vector()).norm("l2") == 0

if __name__ == "__main__":

  # There must be a better way of doing this ...
  import random
  ic = Function(V)
  icvec = ic.vector()
  for i in range(len(icvec)):
    icvec[i] = random.random()

  main(ic)

  adj_html("forward.html", "forward")
  replay_dolfin()
