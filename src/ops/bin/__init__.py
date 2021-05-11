
__all__ = ["clustering_op", "clustering_raw_op", "combinatorial_op", "knn_dataset_op", "rec_dataset_op", "search_op", "tokenizer_op"]

import tensorflow as tf
import os


module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

clustering_op = tf.load_op_library(os.path.join(module_dir,'clustering_op.so'))
clustering_raw_op = tf.load_op_library(os.path.join(module_dir,'clustering_raw_op.so'))
combinatorial_op = tf.load_op_library(os.path.join(module_dir,'combinatorial_op.so'))
knn_dataset_op = tf.load_op_library(os.path.join(module_dir,'knn_dataset_op.so'))
rec_dataset_op = tf.load_op_library(os.path.join(module_dir,'rec_dataset_op.so'))
search_op = tf.load_op_library(os.path.join(module_dir,'search_op.so'))
tokenizer_op = tf.load_op_library(os.path.join(module_dir,'tokenizer_op.so'))


