import tensorflow as tf
import os
import sys
import numpy as np
import logging
assert sys.version_info[0] == 3

import poeem.ops.python.clustering as clustering
import poeem.ops.python.encode as encode


def write_index_file(index_file, codebook, item_id, item_norm, item_code, centroid, assignment, use_residual=True):
    """Write serving related data into a tsv file in the following format
    use_residual               (int8)
    n_batch d n_cluster K D    (int64)
    codebook                   (float32)
    item_id                    (int64)
    item_norm                  (float32)
    item_code                  (uint8 or uint16)
    centroid                   (float32)
    assignment.indices[:, 1]   (uint32)
    """

    with open(index_file, 'wb') as fp:
        use_residual = np.int8(use_residual)
        use_residual.tofile(fp)

        n_batch, D = np.shape(item_code)
        n_cluster, d = np.shape(centroid)
        K = np.shape(codebook)[1]

        dimensions = np.array([n_batch, d, n_cluster, K, D], dtype=np.int64)
        dimensions.tofile(fp)

        codebook.astype(np.float32).tofile(fp)
        item_id.astype(np.int64).tofile(fp)
        item_norm.astype(np.float32).tofile(fp)
        if K <= 256:
            item_code.astype(np.uint8).tofile(fp)
        elif K <= 2**16:  # assume K is at most 2^16
            item_code.astype(np.uint16).tofile(fp)
        else:
            raise ValueError("too large value of K")
        centroid.astype(np.float32).tofile(fp)
        assignment.astype(np.uint32).tofile(fp)


def indexing(data_file, index_file, n_cluster, sample_size, max_iter, change_percentage_thr, use_residual=True):
    """Do indexing for input data and a codebook

    Args:
        data_file: a npz file of a tensor with item_id in [n_batch], item_code in [n_batch, D], codebook in [D, K, d/D]
        K: a scalar indicating number of clusters for coarse quantization
        index_file: output index file

    Returns:
        None
    """
    with open(data_file, 'rb') as fp:
        all_data = np.load(fp, allow_pickle=True)
        item_code = all_data['item_code']
        item_id = all_data['item_id']
        item_norm = all_data['item_norm']
        codebook = all_data['codebook']

    if use_residual:
        # thus, no need to do clustering
        with open(data_file, 'rb') as fp:
            all_data = np.load(fp, allow_pickle=True)
            centroid_val = all_data['coarse_codebook']
            assignment_val = all_data['item_coarse_code']
    else:
        item_code_plh = tf.placeholder(tf.uint16, shape=(None, None), name='item_code_plh')
        codebook_plh = tf.placeholder(tf.float32, shape=(None, None, None), name='codebook_plh')
        centroid, assignment = clustering.kmeans(
            item_code_plh, codebook_plh, n_cluster, sample_size, max_iter, change_percentage_thr)
        with tf.Session() as sess:
            centroid_val, assignment_val = sess.run([centroid, assignment], feed_dict={
                item_code_plh: item_code,
                codebook_plh: codebook
            })
        
    # write the index file
    write_index_file(index_file, codebook, item_id, item_norm, item_code, 
                     centroid_val, assignment_val, use_residual)

    return None
