import backend

def _extract_args(*args, **kwargs):
    eq, u, bcs, _, _, _, _, solver_parameters, _ = backend.solving._extract_args(*args, **kwargs)
    if backend.__name__ == "dolfin":
        if(not isinstance(solver_parameters, dict)):
            solver_parameters = solver_parameters.to_dict()
    return eq, u, bcs, None, None, None, None, solver_parameters
        
        
def _create_random_function(function_space):
    if backend.__name__ == "dolfin":
        r = backend.Function(function_space)
        vec = r.vector()
        for i in range(len(vec)):
            vec[i] = random.random()
        return r
    else:
        r = backend.Function(function_space)
        components = ("((float) rand()) / (float) RAND_MAX",)
        if isinstance(r, backend.Function):
            components *= len(r)
        temp = backend.Expression(components)
        r.interpolate(temp)
        return r
        
if backend.__name__ == "dolfin":
    solve = backend.fem.solving.solve
    matrix_types = lambda: (backend.cpp.Matrix, backend.GenericMatrix)
    _extract_args = backend.fem.solving._extract_args
    function_type = backend.cpp.Function
    function_space_type = backend.cpp.FunctionSpace

else:
    solve = backend.solving.solve
    matrix_types = lambda: backend.op2.petsc_base.Mat
    function_type = backend.Function
    function_space_type = (backend.FunctionSpace, backend.MixedFunctionSpace)

