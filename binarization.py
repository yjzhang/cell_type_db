import numpy as np
from sklearn.cluster import KMeans

# implementations of different strategies for binarizing data
# always assumes that data is of shape (samples, genes)

def binarize_km(data):
    """
    assumes that data is of shape (samples, genes) or (genes,)
    """
    if len(data.shape) == 1:
        km = KMeans(2)
        clusters = km.fit_predict(data.reshape((
                    len(data), 1)))
        return clusters
    elif len(data.shape) == 2:
        km = KMeans(2)
        all_clusters = []
        for i in range(data.shape[1]):
            data_gene = data[:,i]
            clusters = km.fit_predict(data_gene.reshape((
                        len(data_gene), 1)))
            all_clusters.append(clusters)
        return np.array(all_clusters)

def binarize_range(data):
    """
    assumes that data is of shape (samples, genes) or (genes,)
    """
    if len(data.shape) == 1:
        mid = (data.max() - data.min())/2
        new_points = np.zeros(data.shape)
        new_points[data > mid] = 1
        return new_points
    elif len(data.shape) == 2:
        # take max/min of each column
        midpoints = (data.max(1) - data.min(1))/2
        new_points = np.zeros(data.shape)
        for i in range(data.shape[1]):
            new_points[data[:,i] > midpoints[i], i] = 1
        return new_points

def binarize_median(data):
    """
    assumes that data is of shape (samples, genes) or (genes,)
    Assigns data < median to 0, data >= median to 1
    """
    if len(data.shape) == 1:
        median = np.median(data)
        output = np.zeros(data.shape)
        output[data >= median] = 1
        return output
    elif len(data.shape) == 2:
        output = np.zeros(data.shape)
        medians = np.median(data, 0)
        for i in range(data.shape[1]):
            output[data[:,i] >= medians[i], i] = 1
        return output

def binarize_mean(data):
    """
    assumes that data is of shape (samples, genes) or (genes,)
    """
    if len(data.shape) == 1:
        m = np.mean(data)
        output = np.zeros(data.shape)
        output[data >= m] = 1
        return output
    elif len(data.shape) == 2:
        output = np.zeros(data.shape)
        medians = np.mean(data, 0)
        for i in range(data.shape[1]):
            output[:,i][data[:,i] >= medians[i]] = 1
        return output

if __name__ == '__main__':
    pass
