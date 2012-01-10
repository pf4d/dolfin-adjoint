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
