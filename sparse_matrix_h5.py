# Using an LIL matrix and pytables, this is a way to do fast on-disc
# gene lookup for a sparse matrix.

import numpy as np
from scipy import sparse
import tables

def store_matrix(lil_matrix, h5_filename):
    lil_matrix = sparse.lil_matrix(lil_matrix)
    filters = tables.Filters(complevel=5, complib='zlib')
    matrix_file = tables.open_file(h5_filename, mode='w', filters=filters,
            title='matrix')
    data_table = matrix_file.create_vlarray(matrix_file.root,
                    'data', tables.Float64Atom(shape=()),
                    'data',
                    filters=tables.Filters(1))
    for row in lil_matrix.data:
        data_table.append(row)
    rows = matrix_file.create_vlarray(matrix_file.root,
                    'rows', tables.Int64Atom(shape=()),
                    "ragged array of ints",
                    filters=tables.Filters(1))
    for row in lil_matrix.rows:
        rows.append(row)
    shape = matrix_file.create_array(matrix_file.root,
                    'shape', obj=np.array(lil_matrix.shape), title='Matrix shape')
    matrix_file.close()

def to_array(data, row, n_cols):
    array = np.zeros(n_cols)
    array[row] = data
    return array

def load_row(h5_filename, row_number):
    f = tables.open_file(h5_filename, mode='r')
    data_f = f.get_node('/data')
    data = data_f[row_number]
    rows_f = f.get_node('/rows')
    row = rows_f[row_number]
    shape = f.get_node('/shape').read()
    f.close()
    return to_array(data, row, shape[1])

if __name__ == '__main__':
    import scipy.io
    data_mat = scipy.io.loadmat('test_data/10x_pooled_400.mat')
    data = sparse.csc_matrix(data_mat['data'])
    labels = data_mat['labels'].flatten()
    store_matrix(data, 'test.h5')
    row_output = load_row('test.h5', 100)
    assert((row_output == data[100].toarray()).all())
