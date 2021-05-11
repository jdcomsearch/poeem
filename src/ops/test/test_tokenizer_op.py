import tensorflow as tf
import numpy as np

from poeem.ops.python import tokenizer

class TokenizerTest(tf.test.TestCase):
    def testBigramsAndEnTrigramParser(self):
        query = tf.placeholder(tf.string, shape=(None), name='query')
        query_tokens = tokenizer.bigrams_and_en_trigram_parser(query)

        with self.test_session() as sess:
            query_value = np.array(['天气晴abc', '长江test大桥'], dtype=str)
            query_tokens = sess.run(query_tokens, feed_dict={query: query_value})
            query_tokens = [token.decode('utf-8') for token in query_tokens.values]
            print(query_tokens)

            self.assertAllEqual(
                query_tokens,
                ['天', '气', '晴', '#ab', 'abc', 'bc#', '^ 天', '天 气', '气 晴', '晴 #ab', '#ab abc', 'abc bc#', 'bc# $', '长', '江', '#te', 'tes', 'est', 'st#', '大', '桥', '^ 长', '长 江', '江 #te', '#te tes', 'tes est', 'est st#', 'st# 大', '大 桥', '桥 $']
                )

    def testUnigramsAndEnTrigramParser(self):
        query = tf.placeholder(tf.string, shape=(None), name='query')
        query_tokens = tokenizer.unigrams_and_en_trigram_parser(query)

        with self.test_session() as sess:
            query_value = np.array(['天气晴abc', '长江test大桥'], dtype=str)
            query_tokens = sess.run(query_tokens, feed_dict={query: query_value})
            query_tokens = [token.decode('utf-8') for token in query_tokens.values]
            print(query_tokens)

            self.assertAllEqual(
                query_tokens,
                ['天', '气', '晴', '#ab', 'abc', 'bc#', '长', '江', '#te', 'tes', 'est', 'st#', '大', '桥']
                )

if __name__ == "__main__":
    tf.test.main()
