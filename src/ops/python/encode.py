import tensorflow as tf
    
def encode(data, codebook):
    """Encode given item embeddings into K-way D-group code with codebook
    
    For each d/D length subvector, find the nearest vector in the sub-codebook, 
    take the index of the nearest vector as the encoded value

    This function is supposed to be used to export item embeddings

    Args:
        data: a tensor with [n_batch, d]
        codebook: a tensor with [D, K, d/D]

    Returns:
        code: a tensor with [n_batch, D]
    """
    n_item, d = tf.shape(data)[0], tf.shape(data)[1]
    D, K, dd = tf.shape(codebook)[0], tf.shape(codebook)[1], tf.shape(codebook)[2]
    with tf.control_dependencies([tf.assert_equal(dd * D, d, message='inputs and codebook must have consistent dimensions')]):
        reshaped_data = tf.reshape(data, [n_item, D, dd])  # (n_batch, D, dd)
        norm_1 = tf.reduce_sum(reshaped_data**2, -1, keep_dims=True)  # (n_batch, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(codebook**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(reshaped_data, perm=[1, 0, 2]),
                        tf.transpose(codebook, perm=[0, 2, 1]))  # (D, n_batch, K)
        distance = norm_1 - 2 * tf.transpose(dot, perm=[1, 0, 2]) + norm_2
        distance = tf.reshape(distance, [-1, D, K])  # (n_batch, D, K)
        code = tf.argmin(distance, axis=-1)  # (n_batch, D)
        return code
