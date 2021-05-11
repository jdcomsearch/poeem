import tensorflow as tf

from poeem.ops.bin import tokenizer_op

def unigrams_and_en_trigram_parser(query):
    """
    Args:
        query: a tensor with [n_batch, ]

    Returns:
        sparse tensor
    """
    return tf.SparseTensor(*tokenizer_op.unigrams_and_en_trigram_parser(query))

def bigrams_and_en_trigram_parser(query):
    """
    Args:
        query: a tensor with [n_batch, ]

    Returns:
        sparse tensor
    """
    return tf.SparseTensor(*tokenizer_op.bigrams_and_en_trigram_parser(query))
