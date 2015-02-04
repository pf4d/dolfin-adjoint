import backend
import numpy

def _extract_args(*args, **kwargs):
    eq, u, bcs, _, _, _, _, solver_parameters, _ = backend.solving._extract_args(*args, **kwargs)
    if backend.__name__ == "dolfin":
        if(not isinstance(solver_parameters, dict)):
            solver_parameters = solver_parameters.to_dict()
    return eq, u, bcs, None, None, None, None, solver_parameters
        
def randomise(x):
    """ Randomises the content of x, where x can be a Function or a numpy.array.
    """
    
    if hasattr(x, "vector"):
       if backend.__name__ == "dolfin":
           vec = x.vector()
           vec_size = vec.local_size()
           vec.set_local(numpy.random.random(vec_size))
           vec.apply("")
       else:
           components = ("((float) rand()) / (float) RAND_MAX",)
           if isinstance(x, backend.Function):
             if(x.rank() > 0):
               components *= len(x)
           temp = backend.Expression(components)
           x.interpolate(temp)
    else:
        # Make sure we get consistent values in MPI environments
        numpy.random.seed(seed=21)
        x[:] = numpy.random.random(len(x))

if backend.__name__ == "dolfin":
    solve = backend.fem.solving.solve
    matrix_types = lambda: (backend.cpp.Matrix, backend.GenericMatrix)
    _extract_args = backend.fem.solving._extract_args
    function_type = backend.cpp.Function
    function_space_type = backend.cpp.FunctionSpace

else:
    solve = backend.solving.solve
    matrix_types = lambda: backend.op2.base.Mat
    function_type = backend.Function
    function_space_type = (backend.FunctionSpace, backend.MixedFunctionSpace)

