import tensorflow as tf
from poeem.ops.bin import combinatorial_op

# this is adpated from tf.contrib.images.bipartite_match to be undirected.
def bipartite_match(distance_mat):
    """
    Args:
        distance_mat: an nxn matrix of distances.

    Returns:
        - indices matched to each item, e.g., [1, 0, 3, 2] means the match is (0, 1), (2, 3).
    """
    return combinatorial_op.undirected_bipartite_match(distance_mat, -1.0)    # -1.0 means all rows.


if __name__ == '__main__':
    print(tf.Session().run(bipartite_match([[1, 2], [3, 4]])))
