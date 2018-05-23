import unittest

import numpy as np
import scipy.io
from scipy import sparse
import uncurl

import query_archs4

class ArchS4Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_query(self):
        # load data
        data_mat = scipy.io.loadmat('test_data/10x_pooled_400.mat')
        data = sparse.csc_matrix(data_mat['data'])
        labels = data_mat['labels'].flatten()
        gene_names = []
        with open('test_data/10x_pooled_400_gene_names.tsv') as f:
            gene_names = [x.strip() for x in f.readlines()]
        gene_subset = uncurl.max_variance_genes(data)
        data_subset = data[gene_subset, :]
        gene_names = np.array(gene_names)
        gene_names_subset = gene_names[gene_subset]
        bulk_means = {}
        for x in set(labels):
            print('label: ' + str(x))
            means = np.array(data_subset[:,labels==x].mean(1)).flatten()
            bulk_means[x] = means
            tissue_lls = query_archs4.query_tissues(means,
                    gene_names_subset, metric='poisson')
            top_tissues = sorted(tissue_lls.items(), key=lambda x: x[1][0],
                    reverse=True)
            print(list(top_tissues)[:10])

if __name__ == '__main__':
    unittest.main()
