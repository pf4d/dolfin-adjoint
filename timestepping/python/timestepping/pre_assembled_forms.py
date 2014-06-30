#!/usr/bin/env python2

# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014 University of Edinburgh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import OrderedDict
import copy

import dolfin
import ufl

from caches import *
from exceptions import *
from fenics_overrides import *
from fenics_utils import *
from statics import *
from versions import *

__all__ = \
  [
    "NonPAFilter",
    "PAFilter",
    "PAForm",
    "PAMatrixFilter",
    "PAStaticFilter"
  ]

if dolfin_version() < (1, 4, 0):
  def preprocess_integral(integral):
    """
    Given an Integral associated with the given Form, return the integrand and
    a list of arguments which can be used to construct an Integral from the
    integrand.
    """
    
    integrand = integral.integrand()
    domain_type, domain_description, compiler_data, domain_data = \
      integral.domain_type(), integral.domain_description(), integral.compiler_data(), integral.domain_data()
    return integrand, [domain_type, domain_description, compiler_data, domain_data]
else:
  def preprocess_integral(integral):
    """
    Given an Integral associated with the given Form, return the integrand and
    a list of arguments which can be used to construct an Integral from the
    integrand.
    """
    
    integrand = integral.integrand()
    integral_type, domain, subdomain_id, metadata, subdomain_data = \
      integral.integral_type(), integral.domain(), integral.subdomain_id(), integral.metadata(), integral.subdomain_data()
    return integrand, [integral_type, domain, subdomain_id, metadata, subdomain_data]

def matrix_optimisation(form):
  """
  Attempt to convert a linear form into the action of a bi-linear form.
  Return a (bi-linear Form, Function) pair on success, and None on failure.
  """

  if not isinstance(form, ufl.form.Form):
    raise InvalidArgumentException("form must be a Form")

  # Find the test function
  args = ufl.algorithms.extract_arguments(form)
  if not len(args) == 1:
    # This is not a linear form
    return None

  # Look for a single non-static Function dependency
  tcs = extract_non_static_coefficients(form)
  if not len(tcs) == 1:
    # Found too many non-static coefficients
    return None
  elif not isinstance(tcs[0], dolfin.Function):
    # The only non-static coefficient is not a Function
    return None
  # Found a single non-static Function dependency
  fn = tcs[0]

  # Look for a static bi-linear form whose action can be used to construct
  # the linear form
  mat_form = derivative(form, fn,
    # Hack to work around an obscure FEniCS bug
    expand = dolfin.MPI.num_processes() == 1 or
      (not is_r0_function_space(args[0].function_space()) and not is_r0_function(fn)))
  if n_non_static_coefficients(mat_form) > 0:
    # The form is non-linear
    return None
  try:
    rhs_form = dolfin.rhs(dolfin.replace(form, {fn:dolfin.TrialFunction(fn.function_space())}))
  except ufl.log.UFLException:
    # The form might be inhomogeneous
    return None
  if not is_empty_form(rhs_form):
    # The form might be inhomogeneous
    return None

  # Success
  return mat_form, fn

class PAFilter(object):
  def __init__(self, quadrature_degree, pre_assembly_parameters):
    if not isinstance(quadrature_degree, int) or quadrature_degree < 0:
      raise InvalidArgumentException("quadrature_degree must be a non-negative integer")
    if not isinstance(pre_assembly_parameters, dolfin.Parameters):
      raise InvalidArgumentException("default_pre_assembly_parameters must be a Parameters")
    
                                                          # No real need to copy here
    self.pre_assembly_parameters = pre_assembly_parameters#.copy()
    self._quadrature_degree = quadrature_degree
    self._form = ufl.form.Form([])
    self._n = 0
    self._rank = None
    
    return
  
  def name(self):
    return self.__class__.__name__
  
  def _add(self, form):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    elif is_empty_form(form):
      return
         
    if self._n == 0:
      self._form = form
      self._n += 1
      self._rank = form_rank(form)
    else:
      rank = form_rank(form)
      if not rank == self._rank:
        raise InvalidArgumentException("Unexpected form rank: %i" % rank)
      self._form += form
      self._n += 1
      
    return
  
  def _cache_assemble(self, form, compress = False):
    return assembly_cache.assemble(form,
      form_compiler_parameters = {"quadrature_degree":self._quadrature_degree},
      compress = compress)
 
  if dolfin_version() < (1, 4, 0):
    def _assemble(self, form, tensor = None):
      if tensor is None:
        return assemble(form,
          form_compiler_parameters = {"quadrature_degree":self._quadrature_degree})
      else:
        return assemble(form, tensor = tensor, reset_sparsity = False,
          form_compiler_parameters = {"quadrature_degree":self._quadrature_degree})
  else:
    def _assemble(self, form, tensor = None):
      return assemble(form, tensor = tensor,
        form_compiler_parameters = {"quadrature_degree":self._quadrature_degree})
  
  def add(self, form):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    elif is_empty_form(form):
      return ufl.form.Form([]), 0
    
    self._add(form)
    
    return form, 1
  
  def n(self):
    return self._n
  
  def rank(self):
    return self._rank
  
  def pre_assemble(self):
    return
  
  def match_tensor(self, tensor = None, copy = False): 
    return tensor
  
  def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
    return None
  
class PAStaticFilter(PAFilter):
  def __init__(self, quadrature_degree, pre_assembly_parameters):
    PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
    self.__pre_assembled = None
 
    return
  
  def add(self, form):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    elif is_empty_form(form):
      return ufl.form.Form([]), 0
    
    if is_static_form(form):
      self._add(form)
      self.__pre_assembled = None
      return ufl.form.Form([]), 1
    else:
      return form, 0
  
  def pre_assemble(self):
    if self._n == 0:
      return
    
    self.__pre_assembled = self._cache_assemble(self._form, compress = self.pre_assembly_parameters["compress_matrices"])
    
    return
  
  def match_tensor(self, tensor = None, copy = False):
    if self.__pre_assembled is None:
      if self._n == 0:
        return None
      else:
        raise StateException("Cannot call match_tensor method when not pre-assembled")
    
    if tensor is None:
      if copy:
        return self.__pre_assembled.copy()
      else:
        return self.__pre_assembled
    else:
      if isinstance(self.__pre_assembled, dolfin.GenericMatrix):
        self.__pre_assembled.axpy(0.0, tensor, False)
        tensor.axpy(0.0, self.__pre_assembled, False)
      return tensor
  
  def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
    if self.__pre_assembled is None:
      if self._n == 0:
        return None
      else:
        raise StateException("Cannot call assemble method when not pre-assembled")
    
    L = self.__pre_assembled
    if tensor is None:
      if copy:
        L = L.copy()
      return L
    else:
      if isinstance(L, dolfin.GenericMatrix) and same_nonzero_pattern:
        tensor.axpy(1.0, L, True)
      else:
        tensor += L
      return tensor
 
class PAMatrixFilter(PAFilter):
  def __init__(self, quadrature_degree, pre_assembly_parameters):
    PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
    self.__L = OrderedDict()
    self.__pre_assembled = None
 
    return
  
  def add(self, form):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    elif is_empty_form(form):
      return ufl.form.Form([]), 0
  
    mform = matrix_optimisation(form)
    if mform is None:
      return form, 0
    mat_form, fn = mform
    
    self._add(form)
    if fn in self.__L:
      self.__L[fn] += mat_form
    else:
      self.__L[fn] = mat_form
    self.__pre_assembled = None
    return ufl.form.Form([]), 1
  
  def pre_assemble(self):
    if self._n == 0:
      return
    
    self.__pre_assembled = OrderedDict()
    for fn in self.__L:
      self.__pre_assembled[fn] = self._cache_assemble(self.__L[fn], compress = self.pre_assembly_parameters["compress_matrices"])
 
    return
  
  def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
    if self.__pre_assembled is None:
      if self._n == 0:
        return None
      else:
        raise StateException("Cannot call assemble method when not pre-assembled")
    
    if tensor is None:
      fn, mat = self.__pre_assembled.items()[0]
      L = mat * fn.vector()
      for fn, mat in self.__pre_assembled.items()[1:]:
        L += mat * fn.vector()
      return L
    else:
      for fn, mat in self.__pre_assembled.items():
        tensor += mat * fn.vector()
      return tensor
  
class NonPAFilter(PAFilter):
  def __init__(self, quadrature_degree, pre_assembly_parameters):
    PAFilter.__init__(self, quadrature_degree, pre_assembly_parameters)
    self.__tensor = None
 
    return
  
  def add(self, form):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    elif is_empty_form(form):
      return ufl.form.Form([]), 0
    
    self._add(form)
    self.__tensor = None
    return ufl.form.Form([]), 1
  
  def match_tensor(self, tensor, copy = False):
    if self._n == 0:
      return tensor
    
    one = dolfin.Constant(1.0)
    form = dolfin.replace(self._form, {c:one for c in ufl.algorithms.extract_coefficients(self._form)})    
    if self._rank == 1:
      tensor = self._assemble(form)
    elif self._rank == 2:
      if self.__tensor is None:
        self.__tensor = self._assemble(form)
      self.__tensor.axpy(0.0, tensor, False)
      tensor.axpy(0.0, self.__tensor, False)
    else:
      raise StateException("Unexpected form rank: %i" % self._rank)
    
    return tensor
  
  def assemble(self, tensor = None, same_nonzero_pattern = False, copy = False):
    if self._n == 0:
      return tensor
    
    if self._rank == 2:
      if self.__tensor is None:
        self.__tensor = L = self._assemble(self._form)
      else:
        L = self._assemble(self._form, tensor = self.__tensor)
      if tensor is None:
        if copy:
          L = L.copy()
        return L
      else:
        if same_nonzero_pattern:
          tensor.axpy(1.0, L, True)
        else:
          tensor += L
        return tensor
    else:
      if tensor is None:
        return self._assemble(self._form)
      else:
        tensor += self._assemble(self._form)
        return tensor

class PAForm(object):
  def __init__(self, form, pre_assembly_parameters = {}):
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    
    rank = form_rank(form)
    if rank == 0:
      default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["integrals"]
    elif rank == 1:
      default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"]
    elif rank == 2:
      default_pre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"]
    else:
      raise InvalidArgumentException("Unexpected form rank: %i" % rank)
    npre_assembly_parameters = default_pre_assembly_parameters.copy()
    npre_assembly_parameters.update(pre_assembly_parameters)
    pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)
    
    pa_filters = [PAStaticFilter]
    if rank == 1 and pre_assembly_parameters["matrix_optimisation"]:
      pa_filters.append(PAMatrixFilter)
    
    self.__form = form
    self.__rank = rank
    self.__quadrature_degree = form_quadrature_degree(form)
    self.__deps = ufl.algorithms.extract_coefficients(form)
    self.__is_static = is_static_form(form)
    self.pre_assembly_parameters = pre_assembly_parameters
    self.__set_optimise(form, pa_filters)
    
    return

  def __set_optimise(self, form, pa_filters):
    pa_filters = [filter(self.__quadrature_degree, self.pre_assembly_parameters) for filter in pa_filters]
    non_pa_filter = NonPAFilter(self.__quadrature_degree, self.pre_assembly_parameters)
    
    for filter in pa_filters:
      form, n_added = filter.add(form)
    if not is_empty_form(form):
      if self.pre_assembly_parameters["term_optimisation"]:
        for integral in form.integrals():
          integrand, iargs = preprocess_integral(integral)
          if self.pre_assembly_parameters["expand_form"]:
            if isinstance(integrand, ufl.algebra.Sum):
              terms = [(term, expand_expr(term)) for term in integrand.operands()]
            else:
              terms = [(integrand, expand_expr(integrand))]
          else:
            if isinstance(integrand, ufl.algebra.Sum):
              terms = [(term, [term]) for term in integrand.operands()]
            else:
              terms = [(integrand, [integrand])]
          for term in terms:
            term_n_pa = 0
            non_pa_terms = []
            for sterm in term[1]:
              sterm = ufl.form.Form([ufl.Integral(sterm, *iargs)])
              for filter in pa_filters:
                sterm, n_added = filter.add(sterm)
                term_n_pa += n_added
              non_pa_terms.append(sterm)
            if term_n_pa == 0:
              term = ufl.form.Form([ufl.Integral(term[0], *iargs)])
              term, n_added = non_pa_filter.add(term)
            else:
              for sterm in non_pa_terms:
                non_pa_filter.add(sterm)
      else:
        form, n_added = non_pa_filter.add(form)
        
    n = {}
    n_pa = 0
    n_non_pa = 0
    filters = [non_pa_filter] + pa_filters
    for filter in copy.copy(filters):
      n_filter = filter.n()
      if isinstance(filter, NonPAFilter):
        n_non_pa += n_filter
      else:
        n_pa += n_filter
      n[filter.name()] = filter.n()
      if filter.n() == 0:
        filters.remove(filter)
      else:
        filter.pre_assemble()
    
    if self.__rank == 2 and len(filters) > 1:
      tensor = filters[0].match_tensor(copy = True)
      for filter in filters[1:]:
        tensor = filter.match_tensor(tensor = tensor, copy = False)
      for filter in filters[:-1]:
        tensor = filter.match_tensor(tensor = tensor, copy = False)
      same_nonzero_pattern = True
    else:
      same_nonzero_pattern = False

    self.__filters = filters
    self.__n = n
    self.__n_pre_assembled = n_pa
    self.__n_non_pre_assembled = n_non_pa
    self.__same_nonzero_pattern = same_nonzero_pattern

    return

  def assemble(self, copy = False):
    if len(self.__filters) == 0:
      raise StateException("Cannot assemble empty form")
    elif len(self.__filters) == 1:
      return self.__filters[0].assemble(copy = copy)
    else:
      tensor = self.__filters[0].assemble(copy = True)
      for filter in self.__filters[1:]:
        tensor = filter.assemble(tensor = tensor, same_nonzero_pattern = self.__same_nonzero_pattern, copy = False)
      return tensor
    
  def rank(self):
    return self.__rank
  
  def is_static(self):
    return self.__is_static
  
  def n(self):
    return self.__n
    
  def n_pre_assembled(self):
    return self.__n_pre_assembled

  def n_non_pre_assembled(self):
    return self.__n_non_pre_assembled

  def dependencies(self, non_symbolic = False):
    return self.__deps
  
_assemble_classes.append(PAForm)