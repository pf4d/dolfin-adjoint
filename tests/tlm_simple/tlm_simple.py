import random

from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 3)
debugging["record_all"] = True

def main(ic, annotate=True):
  u = TrialFunction(V)
  v = TestFunction(V)

  mass = inner(u, v)*dx
  soln = Function(V)

  solve(mass == action(mass, ic), soln, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V)
  soln = main(ic)

  perturbation_direction = Function(V)
  vec = perturbation_direction.vector()
  for i in range(len(vec)):
    vec[i] = random.random()

  ICParam = InitialConditionParameter(ic, perturbation_direction)
  final_tlm = tlm_dolfin(ICParam)
