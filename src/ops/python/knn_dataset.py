import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes

from poeem.ops.bin import knn_dataset_op


class KnnDataset(dataset_ops.Dataset):
    def _apply_options(self):   # otherwise tf1.15 would apply OptimizeDataset.
        return self

    def with_options(self, options):    # needed for MirroredStrategy.
        del options
        return self

    def __init__(self,
                 batch_file='',
                 item_feature_dict_file='',
                 positive_item_column_index = -1,
                 random_negative_item_count = 0):

        # sanity check
        assert len(batch_file) > 0, 'batch_file must be non-empty'
        assert len(item_feature_dict_file) > 0, 'item_feature_dict_file must be non-empty'

        self._batch_file = ops.convert_to_tensor(
            batch_file,
            dtype=dtypes.string,
            name="batch_file")
        self._item_feature_dict_file = ops.convert_to_tensor(
            item_feature_dict_file,
            dtype=dtypes.string,
            name="item_feature_dict_file")
        self._positive_item_column_index = positive_item_column_index
        self._random_negative_item_count = random_negative_item_count
        # Create any input attrs or tensors as members of this class.
        super(KnnDataset, self).__init__()

    def _as_variant_tensor(self):
        return knn_dataset_op.knn_dataset(
            self._batch_file,
            self._item_feature_dict_file,
            self._positive_item_column_index,
            self._random_negative_item_count)

    def _inputs(self):
        return []

    @property
    def output_types(self):
        return tf.string

    @property
    def output_shapes(self):
        return tf.TensorShape([])

    @property
    def output_classes(self):
        return tf.Tensor
