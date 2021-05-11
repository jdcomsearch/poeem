import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable_base
from tensorflow.python.training.tracking import tracking as trackable

from poeem.ops.bin import search_op

def index_from_file(index_file, name=None):
    if index_file is None or len(index_file) == 0:
        raise ValueError("index_file must be specified and must not be empty.")

    with tf.name_scope(name, "string_to_index"):
        with tf.name_scope(None, "index"):
            init = IndexInitializer(index_file)
            index = Index(init)
    return index


class IndexInitializer(trackable_base.Trackable):

    def __init__(self, filename, name=None):
        self._name = name
        self._filename_arg = filename
        self._filename = self._track_trackable(
        trackable.TrackableAsset(filename), "_filename")
    
    @property
    def _shared_name(self):
        shared_name = "index_%s" % (self._filename_arg)
    
    def initialize(self, index):
        with tf.name_scope(self._name, "index_file_init", (index.resource_handle,)):
            filename = tf.convert_to_tensor(
                self._filename, tf.string, name="asset_filepath")
            init_op = search_op.initialize_index_from_file(
                index.resource_handle, filename)
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, init_op)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, filename)
        return init_op


class Index(trackable.TrackableResource):
    def __init__(self, initializer):
        super(Index, self).__init__()
        if isinstance(initializer, trackable_base.Trackable):
            self._initializer = self._track_trackable(initializer, "_initializer")
        with tf.init_scope():
            self._resource_handle = self._create_resource()
            self._init_op = self._initialize()
        
    def _initialize(self):
        return self._initializer.initialize(self)

    def _create_resource(self):
        index_ref = search_op.index()
        self._index_name = index_ref.name.split("/")[-1]
        return index_ref
    
    @property
    def name(self):
        return self._index_name
    
    @property
    def initializer(self):
        return self._init_op

    def search(self, query, n_neighbor, n_probe, metric_type, verbose=True, name=None):
        neighbors, scores = search_op.index_search(
            self.resource_handle, query, n_neighbor, n_probe, metric_type)
        return neighbors, scores
