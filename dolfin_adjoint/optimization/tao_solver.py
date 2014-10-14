from dolfin import as_backend_type
from dolfin_adjoint.parameter import FunctionControl
from optimization_solver import OptimizationSolver
import numpy as np

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

        self.tao_problem = PETSc.TAO().create(PETSc.COMM_WORLD)
        
        self.__build_app_context()
        self.__set_parameters()

    def __build_app_context(self):
        from petsc4py import PETSc
                
        rf = self.problem.reduced_functional

        # Map each control to a PETSc Vec...
        ctrl_vecs = []
        for control in rf.parameter:
            if isinstance(control, Function):
                tmp_vec = as_backend_type(control.vector()).vec()
                
            elif isinstance(control, FunctionControl):
                tmp_fn = Function(control.data())
                tmp_vec = as_backend_type(tmp_fn.vector()).vec()
                
            elif isinstance(control, ConstantControl):
                tmp_fn = Constant(control.data())
                tmp_vec = self.__constant_as_vec()
                
            ctrl_vecs.append(tmp_vec)

        # ...then concatenate
        # Make a Vec with appropriate local/global sizes and copy in each rank's local entries
        lsizes = [ctrl_vec.sizes for ctrl_vec in ctrl_vecs]
        nlocal, nglobal = map(sum, zip(*lsizes))

        param_vec = PETSc.Vec().create(PETSc.COMM_WORLD)
        param_vec.setSizes((nlocal,nglobal))
        param_vec.setFromOptions()
        nvec = 0
        for ctrl_vec in ctrl_vecs:

            # Local range indices
            ostarti, oendi = ctrl_vec.owner_range
            rstarti = ostarti + nvec
            rendi = rstarti + ctrl_vec.local_size
                        
            param_vec.setValues(range(rstarti,rendi), ctrl_vec[ostarti:oendi])
            param_vec.assemble()
            nvec += ctrl_vec.size

        work_vec = param_vec.duplicate()

        self.param_vec = param_vec
        self.work_vec = work_vec

        tmp_ctrl = Function(rf.parameter[0].data())
        self.tmp_ctrl = Function(rf.parameter[0].data())

        # TODO: Remove below.
        tmp_ctrl = Function(rf.parameter[0].data())
        work_vec = as_backend_type(tmp_ctrl.vector()).vec()
        self.tmp_ctrl = tmp_ctrl
        self.work_vec = work_vec
        
        class MatrixFreeHessian(): 
            def mult(self, mat, X, Y):
                work_vec.set(0)
                work_vec.axpy(1, X)
                hes = rf.hessian(tmp_ctrl)[0]
                hes_vec = as_backend_type(hes.vector()).vec()
                Y.set(0)
                Y.axpy(1, hes_vec)

        class AppCtx(object):
            ''' Implements the application context for the TAO solver '''

            def __init__(self):
                # create solution vector
                # Use value of parameter object as initial guess for the optimisation
                self.x = param_vec.copy()
                
                # create Hessian matrix
                self.H = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
                dims = (self.x.local_size, self.x.size)
                self.H.createPython((dims,dims), comm=PETSc.COMM_WORLD)
                hessian_context = MatrixFreeHessian()
                self.H.setPythonContext(hessian_context)
                self.H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
                self.H.setUp()

            def objective(self, tao, x):
                ''' Evaluates the functional for the parameter value x. '''
                work_vec.set(0)
                work_vec.axpy(1, x)

                return rf(tmp_ctrl)

            def gradient(self, tao, x, G):
                ''' Evaluates the gradient for the parameter choice x. '''

                self.objective(tao, x)
                # TODO: Concatenated gradient vector
                gradient = rf.derivative(forget=False)[0]
                gradient_vec = as_backend_type(gradient.vector()).vec()

                G.set(0)
                G.axpy(1, gradient_vec)

            def objective_and_gradient(self, tao, x, G):
                ''' Evaluates the functional and gradient for the parameter choice x. '''

                j = self.objective(tao, x)
                self.gradient(tao, x, G)

                return j

            def hessian(self, tao, x, H, HP):
                ''' Updates the Hessian. '''
                
                print "In hessian user action routine"
                print "Updating Hessian: %s" % self.stats(x)
                self.objective(tao, x)

            def stats(self, x):
                return "(min, max): (%s, %s)" % (x.min()[-1], x.max()[-1])

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

    def __constant_as_vec(self, cons):
        """Return a PETSc Vec representing the supplied Constant"""
        
        cvec = PETSc.Vec().create(PETSc.COMM_WORLD)
        
        # Scalar case e.g. Constant(0.1)
        if cons.shape() == ():
            cvec.setSizes((PETSc.DECIDE,1))
            cvec.setFromOptions()
            cvec.set(float(cons))

        # Vector case e.g. Constant((0.1,0.1))
        else:
            vec_length = cons.shape()[0]
            cvec.setSizes((PETSc.DECIDE,vec_length))
            cvec.setFromOptions()

            # Can't iterate over vector calling float on each entry
            # Instead evaluate with Numpy arrays of appropriate length
            # See FEniCS Q&A #592
            vals = np.zeros(vec_length)
            cons.eval(vals, np.zeros(vec_length))

            ostarti, oendi = cvec.owner_range
            for i in range(ostarti, oendi):
                cvec.setValue(i,vals[i])

        cvec.assemble()    
        return cvec

    def solve(self):
        self.tao_problem.solve(self.__user.x)
        sol_vec = self.tao_problem.getSolution()
        self.work_vec.set(0)
        self.work_vec.axpy(1, sol_vec)
        return self.tmp_ctrl
