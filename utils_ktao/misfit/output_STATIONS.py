#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""output STATIONS file for SEM in REF_ENU
"""
import sys
from misfit_semlocal import Misfit

# read command line args
misfit_file = str(sys.argv[1])
out_file = str(sys.argv[2])

print("\n====== initialize\n")
misfit = Misfit()

print("\n====== load data\n")
misfit.load(misfit_file)

print("\n====== output STATIONS\n")
misfit.output_STATIONS(out_file)
