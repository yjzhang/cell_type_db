# tools for working with ARCHS4 data
import pickle

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

sample_series = f.get_node('/meta/Sample_series_id').read()

expression = f.get_node('/data/expression')

# TODO: have a hierarchical lookup strategy:
# 1. For each tissue type, calculate the average expression
# 2. For a given cell, query each of the tissue types, and find the one with the highest Poisson log-likelihood
# 3. After that, find the fine-grained dataset.

# map of tissue names to IDs
expression_tissues = {}
for i, tissue in tqdm(enumerate(sample_tissues)):
    if tissue in expression_tissues:
        expression_tissues[tissue].append(i)
    else:
        expression_tissues[tissue] = [i]
expression_tissue_arrays = {}
for tissue in expression_tissues:
    expression_tissue_arrays[tissue] = np.array(expression_tissues[tissue])

# map of tissue names to means
tissue_means = {}
for i, row in tqdm(enumerate(expression.iterrows())):
    tissue = sample_tissues[i]
    if tissue in tissue_means:
        tissue_means[tissue] += row
    else:
        tissue_means[tissue] = row.astype(float)

for tissue in tissue_means:
    tissue_means[tissue] /= len(expression_tissue_arrays[tissue])

tissues = sorted(list(set(sample_tissues)))
tissue_means_array = np.zeros((len(tissues), len(gene_names)))
for i, t in enumerate(tissues):
    tissue_means_array[i,:] = tissue_means[t]


# write to another hdf5 file
filters = tables.Filters(complevel=5, complib='zlib')

tissue_means_file = tables.open_file('archs4/tissue_means.h5', mode='w',
        title='Tissue Means', filters=filters)

tissue_table = tissue_means_file.create_carray(tissue_means_file.root,
        'means', obj=tissue_means_array, title='Tissue means')

tissue_names_table = tissue_means_file.create_carray(tissue_means_file.root,
        'tissues', obj=np.array(tissues), title='Tissue names')

gene_names_table = tissue_means_file.create_carray(tissue_means_file.root,
        'genes', obj=np.array(gene_names), title='Gene names')

tissue_means_file.close()

# TODO: create some sort of index to accelerate nearest-neighbors lookups? Using nmslib?
