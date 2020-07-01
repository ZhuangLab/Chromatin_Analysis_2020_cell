#Author: Bogdan Bintu
#bbintu@fas.harvard.edu
#Copyright Presidents and Fellows of Harvard College, 2018.

import numpy as np
import sys,os,glob
import StormAnalysisAdditions as saa
if __name__ == "__main__":
    dax_file = sys.argv[1]
    bin_file = sys.argv[2]
    parms = saa.loadParms(sys.argv[3])
    over_write = int(sys.argv[4])==1
    saa.fitFrame(dax_file,bin_file,parms,over_write=over_write)