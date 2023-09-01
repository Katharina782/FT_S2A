import itertools
import collections
import random
import re
import glob
import math
import os

import pyranges as pr
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# length of sequence which enformer gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 393216
# length of central sequence which enformer actually sees (1536 bins)
# ─────┆═════┆════════════════════════┆═════┆─────
SEEN_SEQUENCE_LENGTH = 1536*128
# length of central sequence for which enformer gives predictions (896 bins)
# ─────┆─────┆════════════════════════┆─────┆─────
PRED_SEQUENCE_LENGTH = 896*128


"""Code to handle intervals"""

def position_to_bin(pos, 
                    pos_type="relative",
                    target_interval=None):
    if pos_type == "absolute":
        pos = pos - target_interval.start
    elif pos_type == "relative_padded":
        pos = pos - PADDING
    return pos//128

def bin_to_position(bin,
                    pos_type="relative",
                    target_interval=None):
    pos = bin*128 + 64
    if pos_type == "absolute":
        pos = pos + target_interval.start
    elif pos_type == "relative_padded":
        pos = pos + PADDING
    return pos



"""Code to handle inserting sequences into the genome"""

def extract_refseq_centred_at_landmark(landmark_interval, 
                                       fasta_extractor, 
                                       shift_five_end=0,
                                       rev_comp=False):
    return seq_utils.extract_refseq_centred_at_landmark(landmark_interval,
                                                        fasta_extractor,
                                                        shift_five_end=shift_five_end,
                                                        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                        PADDING=PADDING,
                                                        binsize=128,
                                                        rev_comp=rev_comp)

def insert_variant_centred_on_tss(tss_interval,
                                  variant,
                                  allele,
                                  fasta_extractor, 
                                  shift_five_end=0,
                                  rev_comp=False):
    return seq_utils.insert_variant_centred_on_tss(tss_interval,
                                                   variant,
                                                   allele,
                                                   fasta_extractor,
                                                   shift_five_end=shift_five_end,
                                                   SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                   PADDING=PADDING,
                                                   binsize=128,
                                                   rev_comp=rev_comp)
    
def pad_sequence(insert,
                shift_five_end=0,
                landmark=0,
                rev_comp=False):
    return seq_utils.pad_sequence(insert,
                                shift_five_end=shift_five_end,
                                SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                PADDING=PADDING,
                                binsize=128,
                                landmark=landmark,
                                rev_comp=rev_comp)

def insert_sequence_at_landing_pad(insert,
                                   lp_interval,
                                   fasta_extractor,
                                   mode="center",
                                   shift_five_end=0,
                                   landmark=0,
                                   rev_comp=False,
                                   shuffle=False):
    return seq_utils.insert_sequence_at_landing_pad(insert,
                                                    lp_interval,
                                                    fasta_extractor,
                                                    mode=mode,
                                                    shift_five_end=shift_five_end,
                                                    SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                    PADDING=PADDING,
                                                    binsize=128,
                                                    landmark=landmark,
                                                    rev_comp=rev_comp,
                                                    shuffle=shuffle)
    