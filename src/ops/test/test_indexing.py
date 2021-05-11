import tensorflow as tf
import os
from poeem.ops.python import indexing


class IndexingTest(tf.test.TestCase):
    def testBasic(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_file = os.path.join(dir_path, 'testdata/predict.npz')
        index_file = os.path.join(dir_path, 'testdata/index.idx')
        n_cluster = 8192 
        sample_size = 400000
        max_iter = 100
        centroid_change_ratio_thr = 0.00001
        indexing.indexing(data_file, index_file, n_cluster, sample_size, max_iter, centroid_change_ratio_thr)

if __name__ == "__main__":
    tf.test.main()
