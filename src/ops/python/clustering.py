import tensorflow as tf
from poeem.ops.bin import clustering_op
from poeem.ops.bin import clustering_raw_op


def kmeans(code, codebook, n_cluster, sample_size, max_iter=100, change_percentage_thr=0.0001):
    """Do clustering for given item embeddings into K clusters

    Args:
        code: a tensor with [n_batch, D]
        codebook: a tensor with [D, K, d/D]
        n_cluster: a scalar indicates how many clusters

    Returns:
        centroid: a tensor with [K, d]
        assignment: a tensor with [n_batch]
    """
    centroid, assignment = clustering_op.clustering(
        code, codebook, n_cluster, sample_size, max_iter, change_percentage_thr)
    return centroid, assignment


def kmeans_raw(data, n_cluster, max_iter=100, change_percentage_thr=0.0001, verbose=2):
    """Do clustering for given item embeddings into K clusters

    Args:
        data: a tensor with [n_batch, d]
        n_cluster: a scalar indicates how many clusters

    Returns:
        centroid: a tensor with [K, d]
        assignment: a tensor with [n_batch]
    """
    centroid, assignment = clustering_raw_op.clustering_raw(
        data, n_cluster, max_iter, change_percentage_thr, verbose=verbose)
    return centroid, assignment
