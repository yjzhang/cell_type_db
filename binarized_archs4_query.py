
"""
1. Normalize all datasets so that they sum to 1. 
2. Binarize the normalized expression level for each gene using the qualNorm binarization method - clustering into two clusters and separating by means.
3. Build a search index by Hamming distance. 
"""

import pickle

import numpy as np
import tables
from scipy import sparse
from tqdm import tqdm

import uncurl
from uncurl_analysis import bulk_data

ARCHS4_FILENAME = 'archs4/human_matrix.h5'


filename = 'archs4/human_matrix.h5'

f = tables.open_file(filename, 'r')

genes = f.get_node('/meta/genes')
gene_names = genes.read()

sample_titles = f.get_node('/meta/Sample_title')
sample_titles = sample_titles.read()

# sample_source_names indicates the tissue type
sample_tissues = f.get_node('/meta/Sample_source_name_ch1').read()

sample_geo = f.get_node('/meta/Sample_geo_accession').read()

sample_series = f.get_node('/meta/Sample_series_id').read()

expression = f.get_node('/data/expression')

