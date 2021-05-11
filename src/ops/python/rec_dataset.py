import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes

from poeem.ops.bin import rec_dataset_op

class RecDataset(dataset_ops.Dataset):
    def _apply_options(self):   # otherwise tf1.15 would apply OptimizeDataset.
        return self

    def __init__(self,
                 train_file = "",
                 user_file = "",
                 item_file = "",
                 user_column_index = 0,
                 item_column_index = 1,
                 neg_item_count = 0):
        self.train_file = ops.convert_to_tensor(
            train_file,
            dtype=dtypes.string,
            name="train_file")
        self.user_file = ops.convert_to_tensor(
            user_file,
            dtype=dtypes.string,
            name="user_file")
        self.item_file = ops.convert_to_tensor(
            item_file,
            dtype=dtypes.string,
            name="item_file")
        self.user_column_index = ops.convert_to_tensor(
            user_column_index,
            dtype=dtypes.int32,
            name="user_column_index")
        self.item_column_index = ops.convert_to_tensor(
            item_column_index,
            dtype=dtypes.int32,
            name="item_column_index")
        self.neg_item_count = ops.convert_to_tensor(
            neg_item_count,
            dtype=dtypes.int32,
            name="neg_item_count")

        # Create any input attrs or tensors as members of this class.
        super(RecDataset, self).__init__()

    def _as_variant_tensor(self):
        return rec_dataset_op.rec_dataset(
            self.train_file,
            self.user_file,
            self.item_file,
            self.user_column_index,
            self.item_column_index,
            self.neg_item_count)

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
