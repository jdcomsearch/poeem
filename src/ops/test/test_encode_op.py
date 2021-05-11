#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from poeem.ops.python import encode



class EncodeTest(tf.test.TestCase):
    def testBasic(self):
        with self.test_session():
            item_emb = tf.constant([
                [0.1, 0.4, -0.51, -0.9],
                [0.2, 0.4, -0.2, -0.2],
                [0.1, 0.7, -0.4, -0.8],
                [0.6, 0.4, -0.8, -0.3],
                [0.9, 0.6, -0.2, -0.3]])
            codebook = tf.constant([
                [[0.0, 0.0],
                 [0.0, 1.0],
                 [1.0, 0.0],
                 [1.0, 1.0]],
                [[0.0, 0.0],
                 [0.0, -1.0],
                 [-1.0, 0.0],
                 [-1.0, -1.0]]])
            code = encode.encode(item_emb, codebook)
            self.assertAllEqual(code.eval(), [[0, 3], [0, 0], [1, 1], [2, 2], [3, 0]])


if __name__ == "__main__":
    tf.test.main()
