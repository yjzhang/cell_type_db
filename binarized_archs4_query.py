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
import binarization
binarize = binarization.binarize_range
method_name = 'range'

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

# load binarized ARCHS4 profiles
binary_profiles = np.load('binary_profiles_{0}.npz'.format(method_name))
binary_profiles = binary_profiles['binary_profiles']
binary_profiles = binary_profiles.astype(int)

# load query data
# load test data
import scipy.io
data_mat = scipy.io.loadmat('Zeisel/GSE60361_dat.mat')
data = sparse.csc_matrix(data_mat['Dat'])
labels = data_mat['ActLabs'].flatten()
data_gene_names = []
with open('Zeisel/genes.txt') as f:
    data_gene_names = [x.strip().upper() for x in f.readlines()]

# data gene selection
gene_subset = uncurl.max_variance_genes(data, 1, 0.3)
print(len(gene_subset))
data_gene_names = np.array(data_gene_names)
data_gene_names = data_gene_names[sorted(gene_subset)]

# calculate overlap of gene names with test input data
overlapping_gene_names = set(gene_names).intersection(set(data_gene_names))
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
n_genes = len(overlapping_gene_names)
print('number of overlapping genes: ' + str(n_genes))
print(overlapping_gene_names)

# take subset of binary_profiles 
sub_binary_profiles = binary_profiles[:, db_gene_indices]
sub_data = data[data_gene_indices, :].T

# function for converting input array to string arrays?
def to_string_array(array):
    """
    Converts a dense array to a string array?
    """
    output = []
    for i in range(array.shape[0]):
        output.append(' '.join([str(k) for k in array[i,:]]))
    return output

profiles_str = to_string_array(sub_binary_profiles)

# TODO: buid nmslib index
import nmslib
index = nmslib.init(method='hnsw', space='bit_hamming',
        data_type=nmslib.DataType.OBJECT_AS_STRING,
        dtype=nmslib.DistType.INT)
index.addDataPointBatch(profiles_str)
#for i in range(sub_binary_profiles.shape[0]):
#    index.addDataPoint(i, sub_binary_profiles[i,:].astype(np.float32))
index.createIndex()
index.saveIndex('archs4/nmslib_zeisel_genes_bit_hamming_index_{0}'.format(method_name))
print('finished building index')

# TODO: do querying on binarized data
# binarize data, do query
sub_data_binarized = [binarize(sub_data[:,i].toarray()) for i in range(sub_data.shape[0])]
sub_data_binarized = np.array(sub_data_binarized)
sub_data_str = to_string_array(sub_data_binarized)
results = index.knnQueryBatch(sub_data_str, k=5)
result_indices = []
result_distances = []
result_tissues = []
for i in range(sub_data_binarized.shape[0]):
    ind_dist = results[i]
    #cell_data = sub_data_binarized[i,:]
    #query for top 5 cells
    #ind, dist = index.knnQuery(cell_data, 5)
    result_indices.append(ind)
    result_distances.append(dist)
    # top 5 tissues?
    result_tissue = sample_tissues[ind]
    print(result_tissue)
    result_tissues.append(result_tissue)


# save query results
with open('binarized_archs4_query_result_indices.pkl', 'wb') as f:
    pickle.dump(result_indices, f)
with open('binarized_archs4_query_result_distances.pkl', 'wb') as f:
    pickle.dump(result_distances, f)

# TODO: validate results: print some info?
