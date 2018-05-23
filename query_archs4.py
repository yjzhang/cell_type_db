import pickle

import numpy as np
import tables
from scipy import sparse
from tqdm import tqdm

import uncurl
from uncurl_analysis import bulk_data

ARCHS4_FILENAME = 'archs4/human_matrix.h5'
TISSUES_FILENAME = 'archs4/tissue_means.h5'

def query_tissues(data, gene_names, metric='poisson'):
    """
    Args:
        data (array): 1d or 2d array of shape (genes, k) - same shape as M output
            from uncurl.
        gene_names (list): list of strings

    Returns:
        tissue_lls: {tissue_name : [ll for each cell type]}
    """
    tissue_file = tables.open_file('archs4/tissue_means.h5', mode='r')
    means = tissue_file.get_node('/means')
    db_gene_names = tissue_file.get_node('/genes').read()
    tissue_names = tissue_file.get_node('/tissues').read()
    # match genes between the db and the input data
    db_genes_set = set(db_gene_names)
    genes_intersection = db_genes_set.intersection(gene_names)
    # map of gene names to IDs
    db_gene_ids_map = {}
    for i, gene in enumerate(db_gene_names):
        db_gene_ids_map[gene] = i
    data_gene_ids_map = {}
    for i, gene in enumerate(gene_names):
        data_gene_ids_map[gene] = i
    db_gene_ids = []
    data_gene_ids = []
    # TODO: gene subset selection?
    for gene in sorted(list(genes_intersection)):
        db_gene_ids.append(db_gene_ids_map[gene])
        data_gene_ids.append(data_gene_ids_map[gene])
    db_gene_ids = np.array(db_gene_ids)
    data_gene_ids = np.array(data_gene_ids)
    print('overlapping genes: ' + str(len(db_gene_ids)))
    # try linear scan for now...
    if len(data.shape) == 2:
        data_subset = data[data_gene_ids, :]
    elif len(data.shape) == 1:
        data_subset = data[data_gene_ids]
    tissue_lls = {}
    for i, row in enumerate(means.iterrows()):
        tissue_name = tissue_names[i]
        row_normed = row[db_gene_ids]
        row_normed /= row_normed.sum()
        tissue_lls[tissue_name] = []
        if len(data.shape) == 2:
            for k in range(data.shape[1]):
                ll = bulk_data.bulk_query(row_normed, data_subset[:,k], metric)
                tissue_lls[tissue_name].append(ll)
        elif len(data.shape) == 1:
            ll = bulk_data.bulk_query(row_normed, data_subset, metric)
            tissue_lls[tissue_name].append(ll)
    tissue_file.close()
    return tissue_lls

def query_samples(data, gene_name, metric='poisson'):
    pass

def query_samples_nmslib(data, gene_names, metric='cosinesimil'):
    """
    Uses nmslib to query the dataset (first building an index)
    """
