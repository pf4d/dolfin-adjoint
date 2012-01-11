#!/usr/bin/env python

import os, os.path
import sys
import subprocess

tests = {'burgers': 'burgers_picard'}

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]

os.putenv('PYTHONPATH', os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))

for subdir in sorted(subdirs):
  test_name = tests.get(subdir, subdir)
  print "--------------------------------------------------------"
  print "Running %s " % subdir
  print "--------------------------------------------------------"
  os.chdir(os.path.join(basedir, subdir))
  exit = os.system('python %s.py' % test_name)
  assert exit == 0
