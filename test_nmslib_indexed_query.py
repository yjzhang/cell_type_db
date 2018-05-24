# builds a nmslib index of all archs4 profiles...

import pickle

import nmslib
import numpy as np
import tables
from scipy import sparse
import scipy.io
from tqdm import tqdm

filename = 'archs4/human_matrix.h5'

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
# calculate variance over the entire expression matrix...
print('loading expression data...')
expression_data = expression.read()
print('calculating variances...')
variances = expression_data.var(0)
max_variance_indices = np.argsort(variances)[::-1]

# load test data
data_mat = scipy.io.loadmat('test_data/10x_pooled_400.mat')
data = sparse.csc_matrix(data_mat['data'])
labels = data_mat['labels'].flatten()
data_gene_names = []
with open('test_data/10x_pooled_400_gene_names.tsv') as f:
    data_gene_names = [x.strip() for x in f.readlines()]


# select the 1000 genes with highest variance
top_genes = max_variance_indices[:20000]
top_gene_names = gene_names[top_genes]

# calculate overlap of gene names with test input data
overlapping_gene_names = set(top_gene_names).intersection(set(data_gene_names))
overlapping_gene_names = list(overlapping_gene_names)
db_gene_indices = []
data_gene_indices = []
db_gene_ids_map = {}
for i, gene in enumerate(gene_names):
    db_gene_ids_map[gene] = i
data_gene_ids_map = {}
for i, gene in enumerate(data_gene_names):
    data_gene_ids_map[gene] = i
for gene_name in overlapping_gene_names:
    db_gene_indices.append(db_gene_ids_map[gene_name])
    data_gene_indices.append(data_gene_ids_map[gene_name])
print('number of genes: ' + str(len(overlapping_gene_names)))


#np.savetxt('archs4/top_genes_indices.txt', top_genes, fmt='%d')
#np.savetxt('archs4/top_genes_names.txt', top_gene_names, fmt='%s')
# TODO: 
#for i, col in tqdm(enumerate(expression.itercols())):
#    variances.append(col.var())

"""
https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf

potential spaces:
    cosinesimil (cosine similarity)
    l1 (l1 distance)
    l2
    lp:p=...
    linf
    bit_hamming
    jsdivslow, jsdivfast, jsdivfastapprox (Jensen-Shannon divergence)
    kldivfast, kldivfastrq (KL divergence, right query)
"""
space = 'l1'
index = nmslib.init(method='hnsw', space=space)

for i in tqdm(range(expression_data.shape[0])):
    row = expression_data[i, db_gene_indices].astype(np.float32)
    row = row/row.sum()
    index.addDataPoint(i, row)

index.createIndex()
index.saveIndex('archs4/nmslib_{0}_index_data_subset'.format(space))

# test some points
print('test on data')
correct_count = 0
incorrect_count = 0
for i in tqdm(range(expression_data.shape[0])):
    row = expression_data[i, db_gene_indices].astype(np.float32)
    row = row/row.sum()
    ind, dist = index.knnQuery(row, 1)
    if ind[0] == i:
        correct_count += 1
    else:
        incorrect_count += 1
print(correct_count, incorrect_count)

# load index
index = index.loadIndex('archs4/nmslib_{0}_index_data_subset'.format(space))


# test on test dataset
labels = data_mat['labels'].flatten()
for x in set(labels):
    print('label: ' + str(x))
    means = np.array(data[:,labels==x].mean(1)).flatten()
    means = means[data_gene_indices]
    means = means/means.sum()
    ind, dist = index.knnQuery(means, 10)
    # get tissues corresponding to indices
    print(ind, dist)
    print([sample_tissues[i] for i in ind])

# test on individual cells rather than cluster means

