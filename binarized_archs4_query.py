
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

def binarize(expression_profile):
    """
    Converts gene expression profile to a binary form using k-means?
    """
    from sklearn.cluster import KMeans
    km = KMeans(2)
    clusters = km.fit_predict(expression_profile.reshape((
        len(expression_profile), 1)))
    return clusters

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

# TODO: normalize gene expression profiles so that they sum to 1
normalized_expression = []
for i, row in expression.iterrows():
    tissue = sample_tissues[i]
    normed_row = row/row.sum()
    normalized_expression.append(normed_row)

# TODO: binarize gene expression profiles.
# do k-means across genes?
# could also do k-means with k=2 to binarize...
normalized_expression = np.array(normalized_expression)
n_profiles, n_genes = normalized_expression.shape
binary_profiles = []
for g in range(n_genes):
    profile = normalized_expression[:,g]
    binary_profile = binarize(profile)
    binary_profiles.append(binary_profile)
