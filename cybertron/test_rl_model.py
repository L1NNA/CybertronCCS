import unittest

import tensorflow as tf

from cybertron.RLModel import MaximumLikelihoodLoss


class RLModelTests(unittest.TestCase):

    def test_maximum_likelihood_loss(self):
        base = tf.reshape(tf.range(5), [5, 1])
        offset = tf.reshape(tf.range(4), [1, 4])
        y_predict = tf.tile(base, [1, 4]) + offset
        self.assertEqual(y_predict.shape, (5, 4))

        y_predict = tf.slice(y_predict, [0, 0], [-1, 1])
        self.assertEqual(y_predict.shape, (5, 1))
        self.assertTrue((y_predict.numpy() == base.numpy()).all())

        y_comp = y_predict == base
        y_comp = tf.cast(y_comp, 'float32')
        ones = tf.ones((5, 1), dtype='float32')
        self.assertTrue((y_comp.numpy() == ones.numpy()).all())

        y_true = tf.ones((5, 4), dtype='int64')
        probs = tf.constant([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype='float32')
        probs = tf.repeat(probs, 3, axis=1)
        y_predict = tf.concat([ones, probs], axis=1)
        self.assertEqual(y_predict.shape, (5, 4))

        def max_likelihood(reward, probabilities):
            expected_result = -1 * tf.math.log(probabilities + 1e-10)
            expected_result = expected_result * reward
            expected_result = tf.reduce_sum(expected_result, axis=1)
            expected_result = tf.reduce_mean(expected_result)
            return expected_result

        max_ll = MaximumLikelihoodLoss(3)
        result = max_ll(y_true, y_predict)
        self.assertEqual(result.numpy(), max_likelihood(ones, probs).numpy())

        y_true = tf.constant([[0], [1], [1], [1], [1]], dtype='int64')
        result = max_ll(tf.tile(y_true, [1, 4]), y_predict)
        y_true = tf.cast(y_true, 'float32')
        self.assertEqual(result.numpy(), max_likelihood(y_true, probs).numpy())

    def test_cos_sim(self):
        base = tf.reshape(tf.range(5000, dtype='float32'), [100, 50])
        cos_sim = tf.keras.layers.Dot(axes=1, normalize=True)
        bs = tf.keras.losses.BinaryCrossentropy()

        value = cos_sim([base, base])
        value = (value + 1) / 2
        value = bs(tf.ones((100, 1)), value)
        self.assertLess(value.numpy(), 1e-5)

        value = cos_sim([base, -base])
        value = (value + 1) / 2
        value = bs(tf.ones((100, 1)), value)
        self.assertGreater(value.numpy(), 10)  # estimated


if __name__ == '__main__':
    unittest.main()

