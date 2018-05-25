import time

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.io
import scipy.stats
from tqdm import tqdm

# load bulk data
bulk_data = pd.read_csv('Zeisel/NeuronalBulk.csv')
bulk_gene_names = bulk_data['gene.symbol']
bulk_column_names = bulk_data.columns[1:]
bulk_expression = bulk_data.iloc[:, 1:].as_matrix()

# TODO: get a "small gene set" - highest variance genes?
# calculate variance over the entire expression matrix...

# load test data
data_mat = scipy.io.loadmat('Zeisel/GSE60361_dat.mat')
data = sparse.csc_matrix(data_mat['Dat'])
labels = data_mat['ActLabs'].flatten()
data_gene_names = []
with open('Zeisel/genes.txt') as f:
    data_gene_names = [x.strip() for x in f.readlines()]

# data gene selection
import uncurl
gene_subset = uncurl.max_variance_genes(data, 1, 1)
data_gene_names = np.array(data_gene_names)
data_gene_names = data_gene_names[sorted(gene_subset)]

# calculate overlap of gene names with test input data
overlapping_gene_names = set(bulk_gene_names).intersection(set(data_gene_names))
overlapping_gene_names = list(overlapping_gene_names)
db_gene_indices = []
data_gene_indices = []
db_gene_ids_map = {}
for i, gene in enumerate(bulk_gene_names):
    db_gene_ids_map[gene] = i
data_gene_ids_map = {}
for i, gene in enumerate(data_gene_names):
    data_gene_ids_map[gene] = i
for gene_name in overlapping_gene_names:
    db_gene_indices.append(db_gene_ids_map[gene_name])
    data_gene_indices.append(data_gene_ids_map[gene_name])
n_genes = len(overlapping_gene_names)
print('number of genes: ' + str(n_genes))

# linear scan using spearman correlation
# test on test dataset
from uncurl_analysis import bulk_data
corr_matrix = []
pearson_matrix = []
poisson_matrix = []
for x in sorted(list(set(labels))):
    print('label: ' + str(x))
    means = np.array(data[:,labels==x].mean(1)).flatten()
    means = means[data_gene_indices]
    means = means/means.sum()
    max_corr = 0.0
    best_id = 0
    t0 = time.time()
    corrs = []
    pcs = []
    lls = []
    print('label: ' + str(x))
    for i in range(bulk_expression.shape[1]):
        be = bulk_expression[db_gene_indices, i]
        corr = scipy.stats.spearmanr(means, be)
        corr = corr[0]
        pearson_corr = scipy.stats.pearsonr(means, be)[0]
        ll = bulk_data.bulk_query(be/be.sum(), means, 'poisson')
        pcs.append(pearson_corr)
        corrs.append(corr)
        lls.append(ll)
        if corr > max_corr:
            max_corr = corr
            best_id = i
    corr_matrix.append(corrs)
    pearson_matrix.append(pcs)
    poisson_matrix.append(lls)
    # get tissues corresponding to indices
    print(best_id, max_corr)
    print(bulk_column_names[best_id])
    print('elapsed time: ' + str(time.time() - t0))

corr_matrix = np.array(corr_matrix)
pearson_matrix = np.array(pearson_matrix)
poisson_matrix = np.array(poisson_matrix)
# plot Spearman correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 10))
sns.heatmap(corr_matrix, xticklabels=bulk_column_names, linewidths=0.5)
plt.xlabel('Bulk dataset')
plt.ylabel('Cluster')
plt.savefig('rank_corr_zeisel_{0}_genes.png'.format(n_genes))

plt.clf()
sns.heatmap(pearson_matrix, xticklabels=bulk_column_names, linewidths=0.5)
plt.xlabel('Bulk dataset')
plt.ylabel('Cluster')
plt.savefig('pearson_zeisel_{0}_genes.png'.format(n_genes))


plt.clf()
sns.heatmap(poisson_matrix, xticklabels=bulk_column_names, linewidths=0.5)
plt.xlabel('Bulk dataset')
plt.ylabel('Cluster')
plt.savefig('poisson_zeisel_{0}_genes.png'.format(n_genes))


# test on individual cells rather than cluster means
"""
for i, lab in enumerate(labels):
    print('label: ' + str(lab))
    cell = data[:,i].toarray().flatten()
    cell = cell[data_gene_indices]
    cell = cell.astype(np.float32)
    cell = cell/cell.sum()
    max_corr = 0.0
    best_id = 0
    for i in range(bulk_expression.shape[1]):
        corr = scipy.stats.spearmanr(cell, bulk_expression[db_gene_indices, i])[0]
        if corr > max_corr:
            max_corr = corr
            best_id = i
    # get tissues corresponding to indices
    print(best_id, max_corr)
    print(bulk_column_names[best_id])
"""
