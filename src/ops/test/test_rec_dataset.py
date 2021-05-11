import tensorflow as tf
import os
from poeem.ops.python import rec_dataset


class RecDataSetTest(tf.test.TestCase):
    def testRecDataSet(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(dir_path, 'testdata')
        rec_data_set = rec_dataset.RecDataset(
            os.path.join(data_dir, 'batch.txt'),
            '',
            os.path.join(data_dir, 'feature_dict.txt'),
            0,
            2,
            neg_item_count=3)
        rec_data_set = rec_data_set.batch(3)
        rec_data_set = rec_data_set.repeat(10)
        iterator = rec_data_set.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        with self.test_session() as sess:
            try:
                for i in range(2):
                    for e in next_element.eval():
                        print(e.decode())
            except tf.errors.OutOfRangeError:
                print('OutOfRangeError')


if __name__ == "__main__":
    tf.test.main()
