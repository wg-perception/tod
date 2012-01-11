#!/usr/bin/env python
import subprocess
import sys
import os
path = os.path.dirname(sys.argv[0])

print 'feature_viewer'
subprocess.check_call(['%s/../apps/feature_viewer'%path,'--help'])
