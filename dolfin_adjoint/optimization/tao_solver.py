from dolfin import as_backend_type
from optimization_solver import OptimizationSolver

from backend import *

class TAOSolver(OptimizationSolver):
    """Use PETSc TAO to solve the given optimization problem."""

    def __init__(self, problem, parameters=None):
        
        try:
            from petsc4py import PETSc
        except:
            raise Exception, "Could not find petsc4py. Please install it."
        try:
            TAO = PETSc.TAO
        except:
            raise Exception, "Your petsc4py version does not support TAO. Please upgrade to petsc4py >= 3.5."

        OptimizationSolver.__init__(self, problem, parameters)

        self.tao_problem = PETSc.TAO().create(PETSc.COMM_SELF)
        
        self.__build_app_context()
        self.__set_parameters()

    def __build_app_context(self):
        from petsc4py import PETSc
                
        rf = self.problem.reduced_functional

        if len(rf.parameter) > 1:
            raise ValueError, "TAO support is currently limited to 1 parameter"
        
        tmp_ctrl = Function(rf.parameter[0].data())
        tmp_ctrl_vec = as_backend_type(tmp_ctrl.vector()).vec()

        self.tmp_ctrl = tmp_ctrl
        self.tmp_ctrl_vec = tmp_ctrl_vec
        
        class MatrixFreeHessian():
            def mult(self, mat, X, Y):
                tmp_ctrl_vec.set(0)
                tmp_ctrl_vec.axpy(1, X)
                hes = rf.hessian(tmp_ctrl)[0]
                hes_vec = as_backend_type(hes.vector()).vec()
                Y.set(0)
                Y.axpy(1, hes_vec)

        class AppCtx(object):
            ''' Implements the application context for the TAO solver '''

            def __init__(self):
                # create solution vector
                param_vec = as_backend_type(rf.parameter[0].data().vector()).vec()
                # Use value of parameter object as initial guess for the optimisation
                self.x = param_vec.duplicate()
                
                # create Hessian matrix
                self.H = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
                N = self.x.size
                self.H.createPython([N,N], comm=PETSc.COMM_WORLD)
                hessian_context = MatrixFreeHessian()
                self.H.setPythonContext(hessian_context)
                self.H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
                self.H.setUp()

            def objective(self, tao, x):
                ''' Evaluates the functional for the parameter value x. '''

                tmp_ctrl_vec.set(0)
                tmp_ctrl_vec.axpy(1, x)

                return rf(tmp_ctrl)

            def gradient(self, tao, x, G):
                ''' Evaluates the gradient for the parameter choice x. '''

                self.objective(tao, x)
                gradient = rf.derivative(forget=False)[0]
                gradient_vec = as_backend_type(gradient.vector()).vec()

                G.set(0)
                G.axpy(1, gradient_vec)

            def objective_and_gradient(self, tao, x, G):
                ''' Evaluates the gradient for the parameter choice x. '''

                j = self.objective(tao, x)
                self.gradient(tao, x, G)
                return j

            def hessian(self, tao, x, H, HP):
                ''' Evaluates the gradient for the parameter choice x. '''
                print "In hessian user action routine"
                self.objective(tao, x)

        # create user application context
        self.__user = AppCtx()

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""

        from petsc4py import PETSc

        OptDB = PETSc.Options(prefix="tao_")

        if self.parameters is not None:
            for param in self.parameters:

                # Support alternate parameter names
                if param == "method":
                    self.parameters["type"] = self.parameters.pop(param)
                    param = "type"

                elif param == "maximum_iterations" or param == "max_iter":
                    self.parameters["max_it"] = self.parameters.pop(param)
                    param = "max_it"
                    
                # Unlike IPOPT and Optizelle solver, this doesn't raise ValueError on unknown option.
                # Presented as a "WARNING!" message following solve attempt.
                OptDB.setValue(param,self.parameters[param])

        self.tao_problem.setFromOptions()
        
        self.tao_problem.setObjectiveGradient(self.__user.objective_and_gradient)
        self.tao_problem.setObjective(self.__user.objective)
        self.tao_problem.setGradient(self.__user.gradient)
        self.tao_problem.setHessian(self.__user.hessian, self.__user.H)
        self.tao_problem.setInitial(self.__user.x)

    def solve(self):
        self.tao_problem.solve(self.__user.x)
        sol_vec = self.tao_problem.getSolution()
        self.tmp_ctrl_vec.set(0)
        self.tmp_ctrl_vec.axpy(1, sol_vec)
        return self.tmp_ctrl
