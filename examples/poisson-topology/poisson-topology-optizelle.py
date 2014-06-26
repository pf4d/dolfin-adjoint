#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Solve example 1 of

@article{gersborg2006,
year={2006},
journal={Structural and Multidisciplinary Optimization},
volume={31},
number={4},
doi={10.1007/s00158-005-0584-3},
title={Topology optimization of heat conduction problems using the finite volume method},
author={Gersborg-Hansen, A. and Bendsøe, M.P. and Sigmund, O.},
pages={251--259},
}

This problem is to minimise the compliance

\int_\Omega f*T + \alpha |a|_H1

subject to the Poisson equation with mixed Dirichlet--Neumann conditions:

-div(k(a) \nabla T) = f   on \Omega
                  T = 0   on \delta \Omega_D
    (k(a) \nabla T) = 0   on \delta \Omega_N

and to the control constraints
               0 <= a <= 1
        \int_\Omega a <= V

where T is the temperature, a is the control (a(x) == 1 means material, a(x) == 0 means no material),
f is a prescribed source term (here the constant 10^-2), k(a) = \epsilon + (1 - \epsilon) * a^p with \epsilon
and p prescribed constants, \alpha is a regularisation term, and V is the volume bound on the control.

Physically, this corresponds to finding the material distribution a(x) that produces the
least heat when the amount of high conduction material is limited.

This example demonstrates how to implement general control constraints, and how to use IPOPT
to solve the optimisation problem.
"""


from dolfin import *
from dolfin_adjoint import *
set_log_level(ERROR)

try:
  import Optizelle
except ImportError:
  info_red("This example depends on Optizelle.")
  raise

parameters["std_out_all_processes"] = False # turn off redundant output in parallel

V = Constant(0.4)      # volume bound on the control
p = Constant(5)        # power used in the solid isotropic material with penalisation (SIMP) rule, to encourage the control solution to attain either 0 or 1
eps = Constant(1.0e-3) # epsilon used in the solid isotropic material with penalisation (SIMP) rule, used to encourage the control solution to attain either 0 or 1
alpha = Constant(1.0e-8) # regularisation coefficient in functional

def k(a):
  """Solid isotropic material with penalisation (SIMP) conductivity rule, equation (11)."""
  return eps + (1 - eps) * a**p

# Define the discrete function spaces
n = 100
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1) # function space for control
P = FunctionSpace(mesh, "CG", 1) # function space for solution

class WestNorth(SubDomain):
  """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""
  def inside(self, x, on_boundary):
    return (x[0] == 0.0 or x[1] == 1.0) and on_boundary

bc = [DirichletBC(P, 0.0, WestNorth())]                 # the Dirichlet BC; the Neumann BC will be implemented implicitly by dropping the surface integral after integration by parts
f = interpolate(Constant(1.0e-2), P, name="SourceTerm") # the volume source term for the PDE

def forward(a):
  """Solve the forward problem for a given material distribution a(x)."""
  T = Function(P, name="Temperature")
  v = TestFunction(P)

  F = inner(grad(v), k(a)*grad(T))*dx - f*v*dx
  solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})

  return T

if __name__ == "__main__":
  a = interpolate(V, A, name="Control") # initial guess for the control; by interpolating V, the initial guess is feasible
  T = forward(a)                        # solve the forward problem once, to build the dolfin-adjoint tape of the forward problem

  # This block shows how to implement a callback that gets executed every time the functional is evaluated
  # (i.e. the forward PDE is solved. This callback outputs each evaluation to VTK format, for visualisation in paraview.
  # Note that this might be called more often than the number of iterations the optimisation algorithm reports,
  # due to line searches.

  # It is also possible to implement callbacks that are executed on every functional derivative evaluation; see the documentation.
  controls = File("output/control_iterations.pvd")
  a_viz = Function(A, name="ControlVisualisation")
  def eval_cb(j, a):
    a_viz.assign(a)
    controls << a_viz

  # Define the reduced functional, compliance with a weak regularisation term on the gradient of the material
  J = Functional(f*T*dx + alpha * inner(grad(a), grad(a))*dx)
  m = SteadyParameter(a)
  # This ReducedFunctional object solves the forward PDE using dolfin-adjoint's tape each time the functional is to be evaluated,
  # and derives and solves the adjoint equation each time the functional gradient is to be evaluated.
  Jhat = ReducedFunctional(J, m, eval_cb=eval_cb)

  # Now configure the constraints on the control.

  # Bound constraints
  lb = 0.0
  ub = 1.0

  # Volume constraints
  class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
    def __init__(self, V):
      self.V  = float(V)

      # The derivative of the constraint g(x) is constant (it is the diagonal of the lumped mass matrix for the control function space), so let's assemble it here once.
      # This is also useful in rapidly calculating the integral each time without re-assembling.
      self.smass  = assemble(TestFunction(A) * Constant(1) * dx)
      self.tmpvec = Function(A)

    def function(self, m):
      self.tmpvec.assign(m)

      # Compute the integral of the control over the domain
      integral = self.smass.inner(self.tmpvec.vector())
      if MPI.rank(mpi_comm_world()) == 0:
        print "Current control integral: ", integral
        print "Maximum of control: ", m.vector().max()
        print "Minimum of control: ", m.vector().min()
      return [self.V - integral]

    def jacobian(self, m):
      return [-self.smass]

    def jacobian_action(self, m, dm, result):
      result[:] = self.smass.inner(-dm.vector())

    def jacobian_adjoint_action(self, m, dp, result):
      result.vector()[:] = -self.smass*dp

    def length(self):
      """Return the number of components in the constraint vector (here, one)."""
      return 1

  problem = MinimizationProblem(Jhat, constraints=VolumeConstraint(V))

  parameters = {
               "maximum_iterations": 5,
               "optizelle_parameters":
                   {
                   "msg_level" : 10,
                   "algorithm_class" : Optizelle.AlgorithmClass.LineSearch,
                   "H_type" : Optizelle.Operators.UserDefined,
                   "dir" : Optizelle.LineSearchDirection.BFGS,
                   "eps_dx": 1.0e-32,
                   "linesearch_iter_max" : 50,
                   "ipm": Optizelle.InteriorPointMethod.PrimalDual
                   }
               }

  solver  = OptizelleSolver(problem, parameters=parameters)
  a_opt   = solver.solve()
  File("output/control_solution.xml.gz") << a_opt

  # Consider this short form
  # parameters["optimization_backend"] = "TAO"
  # a_opt = optimize(problem)


