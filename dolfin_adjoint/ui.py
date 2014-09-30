import backend
from assembly import assemble, assemble_system
from functional import Functional
from parameter import InitialConditionParameter, ScalarParameter, ScalarParameters, TimeConstantParameter, SteadyParameter, Control

from solving import solve, adj_checkpointing, annotate, record
from adjglobals import adj_start_timestep, adj_inc_timestep, adjointer, adj_check_checkpoints, adj_html, adj_reset
from gst import compute_gst, compute_propagator_matrix, perturbed_replay
from utils import convergence_order, DolfinAdjointVariable
from utils import test_initial_condition_adjoint, test_initial_condition_adjoint_cdiff, test_initial_condition_tlm, test_scalar_parameter_adjoint, test_scalar_parameters_adjoint, taylor_test
from drivers import replay_dolfin, compute_adjoint, compute_tlm, compute_gradient, hessian, compute_gradient_tlm

from variational_solver import NonlinearVariationalSolver, NonlinearVariationalProblem, LinearVariationalSolver, LinearVariationalProblem
from projection import project
from function import Function
from interpolation import interpolate
from constant import Constant
from timeforms import dt, TimeMeasure, START_TIME, FINISH_TIME

# Expose PDE-constrained optimization utilities
from optimization.optimization_problem import *
from optimization.optimization_solver import *
from optimization.ipopt_solver import *
from optimization.optizelle_solver import OptizelleSolver

if backend.__name__ == "dolfin":
  from newton_solver import NewtonSolver
  from krylov_solver import KrylovSolver
  from linear_solver import LinearSolver
  from lusolver import LUSolver
  from reduced_functional import ReducedFunctional, replace_parameter_value, replace_tape_value
  from reduced_functional_numpy import ReducedFunctionalNumPy, ReducedFunctionalNumpy
  from optimization.optimization import minimize, maximize, print_optimization_methods, minimise, maximise
  from optimization.multistage_optimization import minimize_multistage
  from optimization.constraints import InequalityConstraint, EqualityConstraint
  from pointintegralsolver import *
  if hasattr(backend, 'FunctionAssigner'):
    from functionassigner import FunctionAssigner
