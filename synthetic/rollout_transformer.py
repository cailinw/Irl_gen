import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class ROLLOUT(object):
    def __init__(self, transformer, update_rate, reward_gamma):
    	self.transformer = transformer
    	self.update_rate = update_rate
    	self.reward_gamma = reward_gamma

    	# TODO: initialize other model parameters here

    	self.sequence_length = transformer.sequence_length

    def get_reward(self, sess, input_x, roll_num, rewarder):
        rewards = []
        for i in range(roll_num):
            for given_num in range(1, self.sequence_length + 1):
                # TODO: Compute rewards here
        rewards = np.transpose(np.array(rewards)) / (1.0 * roll_num)  # batch_size x seq_length
        return rewards

    def generate(self, sess):
        # outputs, output_probs = sess.run([self.gen_x_old, self.gen_o_old])
        # TODO: Implement this
        return outputs, output_probs

    def update_params(self):
        # TODO: Implement this
        pass