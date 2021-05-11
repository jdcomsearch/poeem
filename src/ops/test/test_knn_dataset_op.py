import tensorflow as tf
import os
from poeem.ops.python import knn_dataset

class KnnDataSetTest(tf.test.TestCase):
    def testKnnDataSetBatchRandom(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dir_path, 'testdata/')
        knn_data_set_obj = knn_dataset.KnnDataset(
            batch_file=os.path.join(data_dir, 'batch.txt'),
            item_feature_dict_file=os.path.join(data_dir, 'feature_dict.txt'),
            positive_item_column_index=2,
            random_negative_item_count=1)
        # for run in range(10):
        knn_data_set = knn_data_set_obj.batch(3)
        iterator = knn_data_set.make_one_shot_iterator()
        next_element = iterator.get_next()
        with self.test_session() as sess:
            dataset = next_element.eval()
            dataset = [line.decode('utf-8').split('\t')[:-2] for line in dataset]
            self.assertAllEqual(dataset, [
                ['101', 'Is jd search awesome?', '1', 'yes'],
                ['202', 'Is deep learning fancy?', '2', 'no'],
                ['303', 'How are you?.', '3', 'not sure']
            ]) 


if __name__ == "__main__":
    tf.test.main()
