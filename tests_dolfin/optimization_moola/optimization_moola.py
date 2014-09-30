""" Solves a MMS problem with smooth control """
from dolfin import *
from dolfin_adjoint import *
try:
  import moola
except ImportError:
  import sys
  info_blue("moola bindings unavailable, skipping test")
  sys.exit(0)


dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx 
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

if __name__ == "__main__":

    n = 100
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')
    x = SpatialCoordinate(mesh)

    x = SpatialCoordinate(mesh)

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) 

    J = Functional((inner(u-u_d, u-u_d))*dx*dt[FINISH_TIME])

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation 
    rf = ReducedFunctional(J, Control(m, value=m))
    problem = rf.moola_problem()
    m_moola = moola.DolfinPrimalVector(m)
    
    solver = moola.SteepestDescent(problem, m_moola, options={'jtol': 0, 'gtol': 1e-10, 'maxiter': 1})

    sol = solver.solve()

    m_opt = sol['control'].data

    #assert max(abs(sol["Optimizer"].data + 1./2*np.pi)) < 1e-9
    #assert sol["Number of iterations"] < 50

    #plot(m_opt, interactive=True)

    solve_pde(u, V, m)


    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")

    # Compute the error
    control_error = errornorm(m_analytic, m)
    state_error = errornorm(u_analytic, u)

