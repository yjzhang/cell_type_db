# correlations with bulk data

# method for correlating cell types (that's not rank correlation)?

# each bulk dataset defines a categorical distribution over genes (bulk datasets are normalized so that they sum to 1)

# so P(cell | bulk) = P_multinomial(cell | bulk_genes, cell_read_count)

# or... each bulk dataset defines a Poisson distribution?

# P(cell_i | bulk) = P_poisson(cell_i | \lambda=bulk_i*cell_read_count)
# and assume that each gene is independent...
# so P(cell | bulk) = \prod_i P_poisson(cell_i | bulk_i*cell_read_count)

# what about comparing a cluster (of single cells) mean to a bulk mean? can we use the same Poisson thing?

# TODO: test classification accuracy of this method vs Euclidean distance, correlation, etc.

# now, what if we have a ton of bulk datasets, and we want to query them? can we do some kinda locality-sensitive hash for probability?

# and how do we do gene matching? we have to have a standard for gene names (an ensembl translator?)

# library of bulk datasets

# alternatively, we can convert cell to a distribution and calculate divergence (but what about presence of zeros? do we just take the nonzeros?

import numpy as np
from scipy import sparse

from scipy.special import xlogy, gammaln
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from uncurl_analysis.sparse_bulk_data import csc_log_prob_poisson_no_norm

# TODO: have a sparse, efficient way of calculating these metrics?
def log_prob_poisson(bulk_data, cell, use_norm=False, eps=1e-10):
    """
    Log-probability of cell given bulk, where
    P(cell | bulk) = \prod_i P_poisson(cell_i | bulk_i*cell_read_count)

    Assumes the same genes in both datasets.

    Args:
        bulk_data (array): 1d array
        cell (array): 1d array
    """
    cell_read_count = cell.sum()
    b = bulk_data*cell_read_count + eps
    if sparse.issparse(cell) and not use_norm:
        cell_csc = sparse.csc_matrix(cell).astype(float)
        return csc_log_prob_poisson_no_norm(
                cell_csc.data,
                cell_csc.indices,
                cell_csc.indptr,
                b,
                eps)
    # prob = bulk^cell*e^(-bulk)/cell!
    # log_prob = cell*log(bulk) - bulk - log(cell!)
    if use_norm:
        return (xlogy(cell, b) - b - gammaln(cell)).sum()
    else:
        return (xlogy(cell, b) - b).sum()

def rank_correlation(bulk_dataset, cell):
    return spearmanr(bulk_dataset, cell)[0]

def pearson_correlation(bulk_dataset, cell):
    return pearsonr(bulk_dataset, cell)[0]

def cosine(bulk_dataset, cell):
    if len(bulk_dataset.shape)==1:
        if len(cell.shape)==1:
            return cosine_similarity(bulk_dataset.reshape(1,-1), cell.reshape(1,-1))
        else:
            if cell.shape[1]==1:
                return cosine_similarity(bulk_dataset.reshape(1,-1),
                        cell.T)
            else:
                return cosine_similarity(bulk_dataset.reshape(1,-1), cell)
    else:
        return cosine_similarity(bulk_dataset, cell)

def bulk_lookup(datasets, cell, method='poisson'):
    """
    Returns a list of (dataset, value) pairs sorted by descending value,
    where value indicates similarity between the cell and the dataset.

    Potential metrics:
        - corr/pearson
        - rank_corr/spearman
        - cosine (normalized cosine distance)
        - poisson (log-probability)

    Test NMI results on 10x_400, all genes:
        Poisson: 1.0
        Spearman: 0.97
        Cosine: 0.85
        Pearson: 0.85

    Args:
        bulk_datasets (dict): dict of (name, 1d np array)
        cell (array): 1d array

    Returns:
        list of (bulk_name, similarity_value) sorted in descending
        similarity
    """
    comp_func = None
    if method == 'poisson':
        comp_func = log_prob_poisson
    elif method == 'spearman' or method == 'rank_corr':
        comp_func = rank_correlation
    elif method == 'cosine':
        comp_func = cosine
    elif method == 'corr' or method == 'pearson':
        comp_func = pearson_correlation
    scores = []
    for name, d in datasets.items():
        s = comp_func(d, cell)
        scores.append((name, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
