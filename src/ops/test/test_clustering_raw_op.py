import tensorflow as tf
import numpy as np
from poeem.ops.python import clustering


class ClusteringRawTest(tf.test.TestCase):
    def testBasic(self):
        with self.test_session():
            num_points = 100
            dimensions = 5
            points1 = np.zeros((1, 5)) + np.random.uniform(0.0,
                                                           1.0, [num_points, dimensions])
            points2 = np.array([[10.0, 2.0, 3.0, 0.0, 0.0]]) + \
                np.random.uniform(0.0, 1.0, [num_points, dimensions])
            points3 = np.array([[1.0, 20.0, 2.0, -2.0, -1.0]]) + \
                np.random.uniform(0.0, 1.0, [num_points, dimensions])
            points4 = np.array([[1.0, -20.0, 2.0, -2.0, -1.0]]) + \
                np.random.uniform(0.0, 1.0, [num_points, dimensions])
            points5 = np.array([[-20.0, 2.0, 2.0, -2.0, -1.0]]) + \
                np.random.uniform(0.0, 1.0, [num_points, dimensions])
            points = np.concatenate(
                [points1, points2, points3, points4, points5], axis=0)

            centroids, assignments = clustering.kmeans_raw(
                tf.convert_to_tensor(points, dtype=tf.float32), 5)


            self.assertAllEqual(tf.equal(
                assignments[num_points*0], assignments[num_points*1]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*0], assignments[num_points*2]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*0], assignments[num_points*3]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*0], assignments[num_points*4]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*1], assignments[num_points*2]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*1], assignments[num_points*3]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*1], assignments[num_points*4]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*2], assignments[num_points*3]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*2], assignments[num_points*4]), False)
            self.assertAllEqual(tf.equal(
                assignments[num_points*3], assignments[num_points*4]), False)


if __name__ == "__main__":
    tf.test.main()
