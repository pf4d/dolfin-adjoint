""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

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
            --petsc.tao_nls_ksp_type cg
            --petsc.tao_nls_pc_type riesz
            --petsc.tao_ntr_ksp_type stcg
            --petsc.tao_ntr_pc_type riesz
            --petsc.tao_ntr_init_type constant
            --petsc.tao_lmm_vectors 20
            --petsc.tao_lmm_scale_type riesz
            --petsc.tao_riesz_ksp_type cg
            --petsc.tao_riesz_pc_type gamg
            --petsc.tao_ls_type armijo
           """.split()
            #--petsc.tao_max_it 0   # Should be used with the NLS algorithm to determine the correct number of KSP solves (and carefully checking that the tolerance is reached within one Newton iteration
print "Tao arguments:", tao_args
parameters.parse(tao_args)

# Create mesh, refined in the center
n = 16   # Use n = 4 for random refine
        # Use n = 4, 8, 16, 32 for uniform refinement
        # Use n = 8 for center refine
mesh = UnitSquareMesh(n, 2*n)

def randomly_refine(initial_mesh, ratio_to_refine= .3):
    numpy.random.seed(0)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cell_markers = cf)

def refine_center(mesh, L=0.2):
    cf = CellFunction("bool", mesh)
    subdomain = CompiledSubDomain('std::abs(x[0]-0.5)<'+str(L)+' && std::abs(x[1]-0.5)<'+str(L))
    subdomain.mark(cf, True)
    return refine(mesh, cf)

#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)

#mesh = refine_center(mesh, L=0.4)
#mesh = refine_center(mesh, L=0.35)
#mesh = refine_center(mesh, L=0.3)
#mesh = refine_center(mesh, L=0.25)
#mesh = refine_center(mesh, L=0.2)
#mesh = refine_center(mesh, L=0.15)

#plot(mesh, interactive=True)
#import sys; sys.exit()

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

f = interpolate(Expression("x[0]+x[1]"), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define regularisation parameter
alpha = Constant(1e-6)

# Define the expressions of the analytical solution
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile
f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")
j_analytic = assemble((1./2*(u_analytic-d)**2 + alpha*f_analytic**2)*dx(mesh))

# Define functional of interest and the reduced functional
ctrl_inner_product = "L2"
regularisation_norm = "L2"

if regularisation_norm == "L2":
    J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
elif regularisation_norm == "H1":
    J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(f)**2)*dx + f**2*dx))
elif regularisation_norm == "H0_1":
    J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*(grad(f)**2)*dx)

control = Control(f)
rf = ReducedFunctional(J, control)

# Calculate Riesz map L2
riesz_V = f.function_space()
riesz_u = TrialFunction(riesz_V)
riesz_v = TestFunction(riesz_V)
riesz_map = assemble(inner(riesz_u, riesz_v)*dx)

# Set Riesz map to identity matrix
#riesz_map.zero()
#riesz_map.ident_zeros()

#problem = MinimizationProblem(rf, bounds=(-1.0,1.0))
problem = MinimizationProblem(rf)
#parameters = None
parameters = { "type": "blmvm",
               "max_it": 2000,
               "fatol": 1e-100,
               "frtol": 0.0,
               "gatol": 1e-8,
               "grtol": 0.0
             }
solver = TAOSolver(problem, parameters=parameters, riesz_map=riesz_map)
#solver = TAOSolver(problem, parameters=parameters, riesz_map=None)

f_opt = solver.solve()
File("output/f_opt.pvd") << f_opt
plot(f_opt, interactive=True)
