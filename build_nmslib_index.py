# builds a nmslib index of all archs4 profiles...


import nmslib
import numpy as np
import tables
from tqdm import tqdm

filename = 'archs4/human_matrix_v11.h5'

print('opening archs4 file...')
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
# no... do a differential expression and find the most differentially expressed genes in each cell type.

tissue_means_filename = 'archs4/tissue_means.h5'
tissue_means_table = f.get_node('/means').read()
tissue_names_table = f.get_node('/tissues').read()

# calculate differential expression...

# calculate variance over the entire expression matrix...
print('loading expression data...')
expression_data = expression.read()
print('calculating variances...')
variances = expression_data.var(0)
max_variance_indices = np.argsort(variances)[::-1]

# select the 1000 genes with highest variance
top_genes = max_variance_indices[:1000]
top_gene_names = gene_names[top_genes]
np.savetxt('archs4/top_genes_indices.txt', top_genes, fmt='%d')
np.savetxt('archs4/top_genes_names.txt', top_gene_names, fmt='%s')
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

for i in tqdm(range(expression_data.shape[0])):
    row = expression_data[i,:]
    index.addDataPoint(i, row[top_genes].astype(np.float32))

index.createIndex()
index.saveIndex('archs4/nmslib_cosine_index')

# test some points
correct_count = 0
incorrect_count = 0
for i in tqdm(range(expression_data.shape[0])):
    row = expression_data[i,top_genes]
    ind, dist = index.knnQuery(row, 1)
    if ind[0] == i:
        correct_count += 1
    else:
        incorrect_count += 1
