from dolfin import *
from dolfin_adjoint import *

# A dummy test to check the Hessian testing in the Taylor test. 
# Tests that check tests that check tests!

a = Constant(3.0, name="MyConstant")
af = float(a)

def J(a):
  af = float(a)
  return af**3

dJda = 3*af**2

def HJa(adot):
  return 6*af*adot

Ja = J(a)

minconv = taylor_test(J, Control(a), Ja, dJda, HJa)
assert minconv > 2.9
