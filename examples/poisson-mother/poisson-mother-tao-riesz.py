""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + alpha/2 || f ||^2

    subjecct to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""

from dolfin import *
from dolfin_adjoint import *
import numpy.random

set_log_level(ERROR)
parameters['std_out_all_processes'] = False

tao_args = """
            --petsc.tao_view
            --petsc.tao_monitor
            --petsc.tao_converged_reason
            --petsc.tao_lmm_vectors 20
            --petsc.tao_lmm_scale_type riesz
            --petsc.tao_riesz_ksp_type cg
            --petsc.tao_riesz_pc_type gamg
            --petsc.tao_ls_type gpcg
           """.split()
parameters.parse(tao_args)

# Create base mesh
n = 16
mesh = UnitSquareMesh(n, 2*n)

def randomly_refine(initial_mesh, ratio_to_refine= .3):
    numpy.random.seed(0)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cell_markers = cf)

# To demonstrate mesh independence, try refining cells at random
# (increase the loop counter)

for i in range(2):
    mesh = randomly_refine(mesh)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1) # state space
W = FunctionSpace(mesh, "CG", 1) # control space

f = interpolate(Constant(0), W, name="Control") # zero initial guess
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define regularisation parameter
alpha = Constant(1e-6)

# Define the expressions of the analytical solution -- we choose our data
# so that the control has an analytic expression
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile
f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")
j_analytic = assemble((1./2*(u_analytic-d)**2 + alpha*f_analytic**2)*dx(mesh))

# Define functional of interest and the reduced functional

J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)

control = Control(f)
rf = ReducedFunctional(J, control)

# Calculate Riesz map for L2 (if we had a H1 regularisation term,
# this would be a Helmholtz operator)
riesz_V = f.function_space()
riesz_u = TrialFunction(riesz_V)
riesz_v = TestFunction(riesz_V)
riesz_map = assemble(inner(riesz_u, riesz_v)*dx)

problem = MinimizationProblem(rf, bounds=(0.0,0.9))

# For the problem without bound constraints, uncomment:
#problem = MinimizationProblem(rf)

parameters = { "type": "blmvm",
               "max_it": 2000,
               "fatol": 1e-100,
               "frtol": 0.0,
               "gatol": 1e-9,
               "grtol": 0.0
             }

# Now construct the TAO solver and pass the Riesz map.
solver = TAOSolver(problem, parameters=parameters, riesz_map=riesz_map)

# To see what happens when you disable the Riesz map, uncomment
# this line:
#solver = TAOSolver(problem, parameters=parameters, riesz_map=None)

f_opt = solver.solve()
File("output/f_opt.pvd") << f_opt
plot(f_opt, interactive=True)
