# TODO: Make sure that this is compatible with v1.
import keras
from keras.layers import Dense

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class GeneratorTransformer(object):
    def __init__(self, num_emb, batch_size, sequence_length,
                 start_token, learning_rate=0.005):
        self.network = self.init_network(10)

        self.num_emb = num_emb
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate

    """
    TODO: Edit this to be the actual architecture and hyperparameters.
    """

    def init_network(self, input_shape, num_outputs=10):

        network = keras.Sequential()
        network.add(Dense(input_shape, activation="relu"))
        network.add(Dense(units=num_outputs, activation="softmax"))

        return network

    """
    TODO: The original code takes a `start_token` argument at initialization,
    and runs `_g_recurrence` many times via a built-in while loop function. 
    `g_recurrence` appends a new generated token to a tensor `gen_x`, and the 
    public-facing `generate` takes the tensorflow session as an argument, and 
    just computes the tensor `gen_x`. All of the appends and what not are part
    of the computation graph for `gen_x`.
    """

    def generate(self, sess):
        outputs = sess.run(self.gen_x)

        """
        Dimension of outputs: [batch_size x seq_length]
        Not sure about last dimension, but seems like there a sampling operation and cast to int
        before the token is appended to `gen_x` in the original code, which implies that it
        is a sequence of ints which represent the word number in the vocabulary.
        Confirm this by running on colab. ~ tensor of ints of shape (batch_size, seq_length)
        """
        return outputs

    """
    TODO: This is a step in the following loop:
    for total_batch in range(TOTAL_BATCH):

        # Train the generator for one step
        start = time.time()
        g_losses = []
        off_samples, off_probs = off_policy_samples(sess, rollout, BATCH_SIZE, off_num)
        avg_reward = []
        for g_it in range(1):
            for it in range(off_num // BATCH_SIZE):
                rewards = rollout.get_reward(sess, off_samples[it], 8, rewarder)
                avg_reward.append(rewards)
            baseline = np.zeros(SEQ_LENGTH)
            for it in range(1): (?)
                for it2 in range(off_num // BATCH_SIZE):
                    _, g_loss = generator.rl_train_step(
                        sess,
                        off_samples[it2],
                        avg_reward[it2],
                        baseline,
                        off_probs[it2],
                        entropy_w,
                        G_rate,
                    )
                    g_losses.append(g_loss)
    """

    def rl_train_step(
        self, sess, x, rewards, baseline, offpolicy, decay_weight, learn_rate
    ):
        """
        These tensors are defined in `__init__`.
        choice_a = ratio * (self.rewards - accumlated_pred * self.decay_weight - self.baseline)
        choice_b = clipped_ratio * (self.rewards - accumlated_pred * self.decay_weight - self.baseline)
        self.g_loss = - tf.reduce_mean(tf.minimum(choice_a, choice_b))
        g_opt = self.optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

        Take one step of optimization. Can be done via:
        https://keras.io/guides/writing_a_training_loop_from_scratch/
        """
        outputs = sess.run(
            [self.g_updates, self.g_loss],
            feed_dict={
                self.x: x,
                self.rewards: rewards,
                self.baseline: baseline,
                self.off_policy_prob: offpolicy,
                self.decay_weight: decay_weight,
                self.learning_rate: learn_rate,
            },
        )
        return outputs

    """
    TODO: This just selects the optimizer object. 
    We might not need this, as it is only internally called.
    """

    def optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)
