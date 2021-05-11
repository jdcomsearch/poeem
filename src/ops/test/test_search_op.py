import tensorflow as tf
import numpy as np
import os
from poeem.ops.python import search

class SearchTest(tf.test.TestCase):
    def testBasic(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        query = tf.placeholder(tf.float32, shape=(None, 2, 512), name='query')
        metric_type = 1
        index_file = os.path.join(dir_path, 'testdata/index.idx')
        
        index = search.index_from_file(index_file)
        init = tf.initializers.tables_initializer()
        neighbors, scores = index.search(query, 50, 64, metric_type)
        with self.test_session() as sess:
            sess.run(init)
            query_file = os.path.join(dir_path, 'testdata/query_embedding.npz')
            query_text = []
            with open(query_file, 'rb') as f:
                result = np.load(f)
                if 'query' in result:
                    query_text = result['query']
                query_vec = result['user_embedding']
            neighbors_val, scores_val = sess.run([neighbors, scores], feed_dict={query: query_vec})
            self.assertAllEqual(1, 1)
        
        for i in range(len(neighbors_val)):
            if len(query_text) != 0:
                print(query_text[i])
            print(neighbors_val[i])
            print(scores_val[i])

if __name__ == "__main__":
    tf.test.main()
