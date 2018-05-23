# builds a nmslib index of all archs4 profiles...

import pickle

import nmslib
import numpy as np
import tables
from scipy import sparse
from tqdm import tqdm

filename = 'archs4/human_matrix.h5'

f = tables.open_file(filename, 'r')

genes = f.get_node('/meta/genes')
gene_names = genes.read()

sample_titles = f.get_node('/meta/Sample_title')
sample_titles = sample_titles.read()

# sample_source_names indicates the tissue type
sample_tissues = f.get_node('/meta/Sample_source_name_ch1').read()
sample_geo = f.get_node('/meta/Sample_geo_accession').read()
# this is the one that we need...
sample_series = f.get_node('/meta/Sample_series_id').read()
expression = f.get_node('/data/expression')

# TODO: get a "small gene set" - highest variance genes?
# calculate variance over the entire expression matrix...
expression_data = expression.read()
variances = expression_data.var(0)
max_variance_indices = np.argsort(variances)[::-1]
# select the 1000 genes with highest variance
top_genes = max_variance_indices[:1000]
# TODO: 
#for i, col in tqdm(enumerate(expression.itercols())):
#    variances.append(col.var())

"""
https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf

potential metrics:
    cosinesimil (cosine similarity)
    l1 (l1 distance)
    l2
    lp:p=...
    linf
    bit_hamming
    jsdivslow, jsdivfast, jsdivfastapprox (Jensen-Shannon divergence)
    kldivfast, kldivfastrq (KL divergence, right query)
"""
index = nmslib.init(method='hnsw', space='cosinesimil')

for i, row in tqdm(enumerate(expression.iterrows())):
    index.addDataPoint(row[top_genes])

index.saveIndex('archs4/nmslib_cosine_index')
