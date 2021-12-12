
"""
1. Compute some variant of tf-idf for each gene/cell - each gene is treated as a word, while each sample is treated as a document. Can this be preceeded by the binarization process?
2. Build a search index by cosine distance.

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

new_expression = []

for i, row in tqdm(enumerate(expression.iterrows())):
    pass
