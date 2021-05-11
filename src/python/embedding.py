import tensorflow as tf
from tensorflow.estimator import ModeKeys
import contextlib

from poeem.ops.python import clustering


def compute_distortion(x, x_tau):
    # x, x_tau : [n_batch, d]
    return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(x - x_tau), axis=1)))


def compute_rotation(x, x_tau):
    """
    solve the orthogonal Procrustes problem via closed form solution
    check the paper "Optimized Product Quantization for Approximate Nearest Neighbor Search"
    basically,
    min_R ||RX - Y||_F^2
    s.t.   R^T R = I

    the solution to R can be obtained as follows.
    XY^T = USV^T
    R = VU^T

    The below code is for transposed X and Y

    min_R ||XR - Y||_F^2
    s.t.   R^T R = I

    the solution to R can be obtained as follows.
    X^T Y = USV^T
    R = U V^T
    """
    M = tf.matmul(x, x_tau, transpose_a=True)
    s, u, v = tf.svd(M)
    R = tf.matmul(u, v, transpose_b=True)
    return R


class PoeemQuantizer(object):
  def __init__(self, coarse_K, K, D, d, rotate, buffer_size, svd_steps=200):
    """
    Args:
      coarse_K: int, size of coarse quantization
      K, D: int, size of KD code.
      d: dim of continuous input for each of D axis.
      rotate: int, if we want to learn a rotation to minimize distortion
      buffer_size: data buffer to initialize centroids
    """
    self._coarse_K = coarse_K
    self._K = K
    self._D = D
    self._d = d
    self._sub_d = d // D
    self._buffer_size = buffer_size
    self._rotate = rotate
    self._svd_steps = svd_steps
    assert self._sub_d >= 1, '_sub_d = 0, _sub_d should be greater and equal than 1, this is caused by that D is greater than embedding dimension'

    self._sample_buffer = tf.Variable(
      name='sample_buffer',
      initial_value=tf.zeros([0, self._d]),
      shape=[None, self._d],
      use_resource=True,  # necessary to have undetermined dimension size.
      trainable=False)
    self._sample_size = tf.get_variable(
      name='sample_size',
      shape=[],
      dtype=tf.int32,
      initializer=tf.zeros_initializer(),
      use_resource=True,  # necessary if variable is used in tf.cond condition
      trainable=False)
    self._initialized = tf.get_variable(
      name='initialized',
      shape=[],
      dtype=tf.int8,
      initializer=tf.zeros_initializer(),
      use_resource=True,  # necessary if variable is used in tf.cond condition
      trainable=False)
    self._coarse_centroids = tf.get_variable(
      name='coarse_centroids',
      shape=[self._coarse_K, self._d]) if self._coarse_K > 0 else None
    self._centroids = tf.get_variable(
      name="centroids_k",
      shape=[self._D, self._K, self._sub_d])
    self._rotation_matrix = tf.get_variable(
      name="rotation_matrix",
      shape=[self._d, self._d],
      initializer=tf.keras.initializers.Identity(),
      trainable=True) if rotate else None

  def coarse_l2_distance(self, x, coarse_centroids):
    n_batch = tf.shape(x)[0]
    norm_1 = tf.tile(
        tf.reduce_sum(x**2, axis=-1, keep_dims=True),  # n_batch x coarse_K
        [1, self._coarse_K])
    norm_2 = tf.tile(
        tf.transpose(
            tf.reduce_sum(coarse_centroids**2, axis=-1, keep_dims=True),
            [1, 0]),                                   # n_batch x coarse_K
        [n_batch, 1])
    # x = tf.Print(x, [tf.shape(x), tf.shape(coarse_centroids)], message='before dot ')
    dot = tf.matmul(x, coarse_centroids, transpose_b=True) # n_batch x coarse_K
    # dot = tf.Print(dot, [tf.shape(x), tf.shape(coarse_centroids), tf.shape(dot)], message='after dot ')
    # norm_1 = tf.Print(norm_1, [tf.shape(norm_1), tf.shape(norm_2), tf.shape(dot)], message='before l2_sqr ')
    l2_sqr = norm_1 + norm_2 - 2 * dot                 # n_batch x coarse_K
    # l2_sqr = tf.Print(l2_sqr, [tf.shape(norm_1), tf.shape(norm_2), tf.shape(dot)], message='after l2_sqr ')
    return l2_sqr

  def pq_l2_distance(self, x, centroids):
    n_batch = tf.shape(x)[0]
    x = tf.reshape(x, [n_batch, self._D, self._sub_d])
    norm_1 = tf.reduce_sum(x**2, -1, keep_dims=True)        # (n_batch, D, 1)
    norm_2 = tf.expand_dims(tf.reduce_sum(centroids**2, -1), 0)                         # (1, D, K)
    dot = tf.matmul(tf.transpose(x, perm=[1, 0, 2]),
                    tf.transpose(centroids, perm=[0, 2, 1]))    # (D, n_batch, K)
    l2_sqr = norm_1 + norm_2 - 2 * tf.transpose(dot, perm=[1, 0, 2])  # (n_batch, D, K)
    return l2_sqr

  def quantize(self, x, coarse_centroids, centroids):
    # x: (batch_size, d)
    n_batch = tf.shape(x)[0]
    coarse_code, coarse_output = None, None

    if self._coarse_K > 0:
        # coarse_K > 0 means use residual
        coarse_l2_sqr = self.coarse_l2_distance(x, coarse_centroids)
        coarse_code = tf.argmin(coarse_l2_sqr, -1)
        coarse_output = tf.nn.embedding_lookup(
            coarse_centroids, coarse_code)

        # compute residual
        x = x - coarse_output

    l2_sqr = self.pq_l2_distance(x, centroids)      # (n_batch, D, K)
    code = tf.argmin(l2_sqr, -1)                    # (n_batch, D)
    neighbor_idxs = code

    # Compute the outputs, which has shape (batch_size, D, sub_d)
    D_base = tf.convert_to_tensor(
        [self._K*d for d in range(self._D)], dtype=tf.int64)
    neighbor_idxs += tf.expand_dims(D_base, 0)       # (batch_size, D)
    neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
    centroids = tf.reshape(centroids, [-1, self._sub_d])
    output = tf.nn.embedding_lookup(centroids, neighbor_idxs)
    output = tf.reshape(output, [n_batch, self._d])

    if self._coarse_K > 0:
      x_tau = coarse_output + output
    else:
      x_tau = output

    return x_tau, coarse_code, code

  def compute_centroids(self, x, max_iter=100, change_pct_thr=0.01):
    coarse_centroids = None
    if self._coarse_K > 0:
      coarse_centroids, _ = clustering.kmeans_raw(
        x, self._coarse_K,
        max_iter=max_iter, change_percentage_thr=change_pct_thr, verbose=1)
      coarse_centroids = tf.reshape(coarse_centroids, [self._coarse_K, self._d])

    centroids_list = []
    for i in range(self._D):
      ctrd, _ = clustering.kmeans_raw(
        x[:, i*self._sub_d:(i+1)*self._sub_d], self._K,
        max_iter=max_iter, change_percentage_thr=change_pct_thr, verbose=1)
      centroids_list.append(ctrd)
    centroids = tf.stack(centroids_list, axis=0)
    centroids = tf.reshape(centroids, [self._D, self._K, self._sub_d])
    return coarse_centroids, centroids

  def forward(self, x, rotation_matrix=None):
    """Rotate x, compute quantized  x_tau, then rotate x_tau back."""
    with tf.name_scope("Poeem_forward"):
      if self._rotate > 0:
        assert rotation_matrix is not None
        x = tf.matmul(x, rotation_matrix)
        x_tau, coarse_code, code = self.quantize(
            x, self._coarse_centroids, self._centroids)
        x_tau = tf.matmul(x_tau, tf.transpose(rotation_matrix, [1, 0]))
      else:
        x_tau, coarse_code, code = self.quantize(
            x, self._coarse_centroids, self._centroids)

      return x_tau, coarse_code, code

  def accumulate(self, x):
    # cutoff data to fit in sample buffer
    batch_size = tf.minimum(
      tf.shape(x)[0],
      self._buffer_size - self._sample_size)
    x_sub = tf.slice(x, [0, 0], [batch_size, -1])

    assign_tensor = self._sample_buffer.assign(
      tf.concat([self._sample_buffer, x_sub], axis=0))

    sample_size_tensor = self._sample_size.assign_add(batch_size)
    return assign_tensor, sample_size_tensor

  def init_centroids(self, max_iter=None, change_pct_thr=0.01):
    assign_ops = []
    if self._rotate > 0:
      def loop_body(i, R, data):
        x = tf.matmul(data, R)
        coarse_centroids, centroids = self.compute_centroids(
          x, max_iter=max_iter or 10, change_pct_thr=change_pct_thr)
        x_tau, coarse_code, code = self.quantize(x, coarse_centroids, centroids)
        distortion = compute_distortion(x, x_tau)
        x_tau = tf.Print(x_tau, [i, distortion], message='distortion = ')
        R = compute_rotation(data, x_tau)
        i = tf.add(i, 1)
        return i, R, data

      def condition(i, R, data):
        return tf.less(i, self._svd_steps)

      # find the optimized rotation to minimize distortion, by alternative minimization
      i = tf.constant(0)
      R = tf.eye(self._d)
      data = tf.reshape(self._sample_buffer, [self._buffer_size , self._d])  # necessary to avoid dimension inconsistency error
      _, R, _ = tf.while_loop(
        condition, loop_body, [i, R, data],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([self._d, self._d]),
          tf.TensorShape([self._buffer_size, self._d])])

      # apply the rotation
      x = tf.matmul(self._sample_buffer, R)
      coarse_centroids, centroids = self.compute_centroids(
        x, max_iter=max_iter or 100, change_pct_thr=change_pct_thr)
      assign_ops.append(self._rotation_matrix.assign(R))
    else:
      coarse_centroids, centroids = self.compute_centroids(self._sample_buffer,
        max_iter=max_iter or 100, change_pct_thr=change_pct_thr)
    if self._coarse_K > 0:
      assign_ops.append(self._coarse_centroids.assign(coarse_centroids))
    assign_ops.append(self._centroids.assign(centroids))
    assign_ops.append(self._initialized.assign(1))
    return assign_ops

  def clear_sample_buffer(self):
    buffer_assign = tf.assign(self._sample_buffer, tf.zeros([0, self._d]), validate_shape=False)
    return buffer_assign,


class PoeemEmbed(object):
  def __init__(self, emb_size, warmup_steps=1024, buffer_size=8192, mode=ModeKeys.PREDICT,
      hparams=None, name="poeem_emb", initializer=None, svd_steps=200):
    if hparams is None:
        hparams = PoeemHparam()
    self._warmup_steps = warmup_steps
    self._buffer_size = buffer_size
    self._hparams = hparams
    self._mode = mode
    self._d, self._K, self._D, self._coarse_K, self._rotate = (
      emb_size, hparams.K, hparams.D, hparams.coarse_K, hparams.rotate)
    self._quantizer = PoeemQuantizer(self._coarse_K, self._K, self._D, self._d,
      self._rotate, buffer_size, svd_steps=svd_steps)

  def forward(self, x):
    def forward_layer(x):
      n_batch = tf.shape(x)[0]
      x_tau, coarse_code, code = (
          # the existence of this op is necessary to enable control dependencies working
          tf.identity(x),
          tf.zeros([n_batch], dtype=tf.int64),
          tf.zeros([n_batch, self._D], dtype=tf.int64))
      regularizer = 0.0
      return x_tau, coarse_code, code, regularizer

    def pq_layer(x):
      def accumulate_layer(x):
        with tf.device('/device:CPU:0') if self._hparams.kmeans_on_cpu else (
            contextlib.suppress()):
          deps = self._quantizer.accumulate(x)
          with tf.control_dependencies(deps):
            return forward_layer(x)

      def init_layer(x):
        with tf.device('/device:CPU:0') if self._hparams.kmeans_on_cpu else (
            contextlib.suppress()):
          deps = self._quantizer.init_centroids(
            max_iter=self._hparams.kmeans_max_iter,
            change_pct_thr=self._hparams.kmeans_change_pct_thr)
          with tf.control_dependencies(deps):
            clear_deps = self._quantizer.clear_sample_buffer()
            with tf.control_dependencies(clear_deps):
              return internal_pq_layer(x, init=True)

      def internal_pq_layer(x, init=False):
        rotation_matrix=self._quantizer._rotation_matrix
        if init and rotation_matrix is not None:
          rotation_matrix = tf.stop_gradient(rotation_matrix)
        x_tau, coarse_code, code = self._quantizer.forward(x, rotation_matrix)
        regularizer = tf.reduce_sum((x_tau - tf.stop_gradient(x))**2)
        if coarse_code is None: # dummy placeholder for tf.cond.
          coarse_code = tf.zeros([tf.shape(x)[0]], dtype=tf.int64)
        return x_tau, coarse_code, code, regularizer

      if self._quantizer._buffer_size < 0 or self._mode != ModeKeys.TRAIN:
        return internal_pq_layer(x) # skip kmeans/svd initialization.
      return tf.cond(tf.less(self._quantizer._sample_size, self._quantizer._buffer_size),
        lambda: accumulate_layer(x),    # accumulate data
        lambda: tf.cond(
          tf.equal(self._quantizer._initialized, 0),
          lambda: init_layer(x),        # init centroids
          lambda: internal_pq_layer(x)))

    if self._mode == ModeKeys.TRAIN:
      # forward_layer: 1. steps from 0 to _warmup_steps, train model without pq
      # pq_layer: 2. steps from _warmup_steps to _buffer_size/batch_size,
      #              train model without pq and accumulate input data in _sample_buffer
      #           3. clustering input data in _sample_buffer to init centroids and start to train pq model
      step = tf.cast(tf.train.get_global_step(), tf.int32)
      if self._warmup_steps > 0:
        x_tau, coarse_code, code, regularizer = tf.cond(
          tf.less_equal(step, self._warmup_steps),
          true_fn=lambda: forward_layer(x),
          false_fn=lambda: pq_layer(x))
    else:
      x_tau, coarse_code, code, regularizer = pq_layer(x)

    update_ops = self.metrics(x, x_tau, coarse_code, code)
    with tf.control_dependencies(update_ops):
      x_tau = tf.identity(x_tau)
    return x_tau, coarse_code, code, regularizer

  def metrics(self, x, x_tau, coarse_code, code):
    distortion = compute_distortion(x, x_tau)
    tf.summary.scalar('distortion', distortion)
    update_ops = []
    def full_summary(dim, code, name):
      v, idx, counts = tf.unique_with_counts(code)
      tf.summary.scalar(
        'code_distribution/unique_%s_count' % name, tf.shape(v)[0])
      tf.summary.histogram(
        'code_distribution/%s_histogram' % name, code)
      tally_var = tf.Variable(tf.zeros([dim], dtype=tf.int64),
        name=name + '_cumulative_count', dtype=tf.int64, trainable=False)
      update_ops.append(tf.scatter_add(tally_var, v, tf.to_int64(counts)))
      tally_var = tf.to_float(tally_var)
      cum_frac = (tally_var + 1e-8) / (tf.reduce_sum(tally_var) + 1e-8 * dim)
      kl = tf.keras.losses.KLDivergence()
      # ideally cluster sizes should be uniform, so track distances to uniform.
      uniform = tf.ones([dim], dtype=tf.float32) / dim
      def tv(dist1, dist2): # total variation distance of two prob. measures.
        return tf.reduce_sum(tf.abs(dist1 - dist2)) / 2.0
      tf.summary.scalar('code_distribution/%s_kld' % name, kl(cum_frac, uniform))
      tf.summary.scalar('code_distribution/%s_tvd' % name, tv(cum_frac, uniform))
    if self._coarse_K > 0:
      full_summary(self._coarse_K, coarse_code, 'coarse_code')
    code_part0 = code[:, 0]
    full_summary(self._K, code_part0, 'code0')
    return update_ops


class PoeemHparam(object):
  # A default Poeem parameter setting (demo)
  def __repr__(self):
    return str({k: getattr(self, k) for k in dir(self) if
        not k.startswith('_') and 'built-in' not in str(getattr(self, k))})

  def __init__(self,
               coarse_K=128,
               K=16,
               D=32,
               rotate=0,
               kmeans_max_iter=100,
               kmeans_change_pct_thr=1e-2,
               kmeans_on_cpu=False):
    """
    Args:

    """
    self.coarse_K = coarse_K
    self.K = K
    self.D = D
    self.rotate = rotate
    self.kmeans_max_iter = kmeans_max_iter
    self.kmeans_change_pct_thr = kmeans_change_pct_thr
    self.kmeans_on_cpu = kmeans_on_cpu # gpu may run OOM.