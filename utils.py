import libadjoint
from solving import *

def replay_dolfin():
  if "record_all" not in debugging or debugging["record_all"] is not True:
    print "Warning: your replay test will be much more effective with debugging['record_all'] = True."

  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=0.0)
      storage.set_overwrite(True)
      adjointer.record_variable(fwd_var, storage)

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders

def adjoint_dolfin(functional, forget=True):

  for i in range(adjointer.equation_count)[::-1]:
      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
      
      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      if forget:
        adjointer.forget_adjoint_equation(i)

  return output.data # return the last adjoint state
