
from optparse import OptionParser
import collections
import gzip
import heapq
import pdb
import os
import random
import sys
import time

import numpy as np
import pybedtools



usage = "usage: %prog [options] <file_name1> <file_name2> <output_file> <data_dir>"
parser = OptionParser(usage)
parser.add_option("-l", dest="seq_length",
                  default=196608, type="int",
                  help="Input sequence length [Default: %default]")
parser.add_option("--species", dest="species",
                  default="human",
                  help="Species for which to compute correlations. One of human or mouse.")
parser.add_option("--subset", dest="subset",
                  default="valid",
                  help="One of train, valid or test.")

(options, args) = parser.parse_args()

if len(args) != 4:
    parser.error("Must provide file name, output name, data directory and checkpoint directory")
else:
    bed1 = args[0]#, args[1]]
    bed2 = args[1]
    output_file=args[2]
    data_dir = args[3]
    #checkpoint_dir = args[3]
    

#print(f"seq-len: {options.seq_length}, output_file_extension: {output_file}, species: {options.species}, subset: {options.subset}")


### I would like to use this script to 
# 1. create bed files from input_region file and tss file 
# 2. compute the overlap between the two
# 3. save the overlap_tss and do some data wrangling on it. 

#data_dir = "/omics/groups/OE0540/internal/users/mikulik/master_thesis/data/gcs_basenj"
bed_file1 = '%s/%s' % (data_dir, bed1)
bed_file2 = '%s/%s' % (data_dir, bed2)
output_file = '%s/%s' % (data_dir, output_file)

bed1_open = pybedtools.BedTool(bed_file1)
bed2_open = pybedtools.BedTool(bed_file2)
print(len(bed1_open), len(bed2_open))
overlap_open = open(output_file, "w")


for overlap in bed1_open.intersect(bed2_open, wo=True):
    #print(overlap)
    print(overlap, file=overlap_open)
    
overlap_open.close()
