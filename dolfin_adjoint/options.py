from dolfin import parameters, Parameters

adj_params = Parameters("adjoint")
adj_params.add("record_all", True)
adj_params.add("test_hermitian", False)
adj_params.add("test_derivative", False)
adj_params.add("fussy_replay", True)
adj_params.add("stop_annotating", False)

parameters.add(adj_params)