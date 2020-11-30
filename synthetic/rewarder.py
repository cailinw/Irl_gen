# -*- coding: utf-8 -*-
"""
@author: Samzhanshi
"""
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Rewarder(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token, mid_layer,
                 l2_reg_lambda=0):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        # self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.r_params = []
        self.grad_clip = 5.0
        self.mid_layer = mid_layer
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        with tf.variable_scope('generator'):
            self.r_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.r_params.append(self.r_embeddings)
            self.r_recurrent_unit = self.create_recurrent_unit(self.r_params)  # maps h_tm1 to h_t for generator
            self.r_output_unit = self.create_output_unit(self.r_params, self.mid_layer)  # maps h_t to o_t (output token logits)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator
        self.weight = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.temperature = tf.placeholder(tf.float32, name='temperature')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length]) # get from rollout policy and discriminator

        # processed for batch
        self.pos_weight = tf.nn.softmax(self.weight[:self.batch_size//2])
        self.neg_weight = -1.0 * tf.nn.softmax(self.weight[self.batch_size//2:] / self.temperature)
        self.f_weight = tf.concat([self.pos_weight, self.neg_weight], axis=0)
        with tf.device("/cpu:0"):
            self.word = tf.nn.dropout(tf.nn.embedding_lookup(self.r_embeddings, self.x), self.dropout_keep_prob)
            self.processed_x = tf.transpose(self.word, perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])
        # self.avg_h0 = tf.zeros([self.batch_size, self.hidden_dim])

        gen_h = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)


        # supervised pretraining for generator
        r_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, r_predictions, gen_h):
            gen_h = gen_h.write(i, tf.unstack(h_tm1)[0])
            h_t = self.r_recurrent_unit(x_t, h_tm1)
            o_t = self.r_output_unit(h_t)
            r_predictions = r_predictions.write(i, o_t)  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, r_predictions, gen_h

        _, _, _, self.r_predictions, self.gen_h = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.r_embeddings, self.start_token),
                       self.h0, r_predictions, gen_h))

        # def _pretrain_recurrence(i, x_t, h_tm1, r_predictions, gen_h, avg_h):
        #     gen_h = gen_h.write(i, tf.unstack(h_tm1)[0])
        #     h_t = self.r_recurrent_unit(x_t, h_tm1)
        #     avg_h = (avg_h * i + h_t) / (i + 1)
        #     o_t = self.r_output_unit(avg_h)
        #     r_predictions = r_predictions.write(i, o_t)  # batch x vocab_size
        #     x_tp1 = ta_emb_x.read(i)
        #     return i + 1, x_tp1, h_t, r_predictions, gen_h, avg_h
        #
        # _, _, _, self.r_predictions, self.gen_h = control_flow_ops.while_loop(
        #     cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
        #     body=_pretrain_recurrence,
        #     loop_vars=(tf.constant(0, dtype=tf.int32),
        #                tf.nn.embedding_lookup(self.r_embeddings, self.start_token),
        #                self.h0, r_predictions, gen_h))


        self.r_predictions = tf.transpose(self.r_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # clip_reward & log_pred :  batch*seq  x vocab_size
        self.clipped_reward = tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * \
            tf.clip_by_value(tf.reshape(self.r_predictions, [-1, self.num_emb]), 1e-20, 1.0)
        self.reward_per_step_snt = tf.reshape(tf.reduce_sum(self.clipped_reward, -1), [self.batch_size, self.sequence_length])
        self.sent_reward = tf.reduce_sum(self.reward_per_step_snt, axis=1)
        self.reward_loss = -tf.reduce_sum(self.sent_reward * self.f_weight) + \
                            l2_reg_lambda * (tf.add_n([tf.nn.l2_loss(var) for var in self.r_params if var not in [self.r_embeddings]]))

        reward_opt = self.optimizer(self.learning_rate)

        self.reward_grad, _ = tf.clip_by_global_norm(tf.gradients(self.reward_loss, self.r_params), self.grad_clip)
        self.reward_updates = reward_opt.apply_gradients(zip(self.reward_grad, self.r_params))

    def reward_train_step(self, sess, x, weight, temperature, dropkeep, lr_rate):
        outputs = sess.run([self.reward_updates, self.reward_loss], feed_dict={self.x: x, self.weight: weight, self.temperature: temperature,
                    self.dropout_keep_prob: dropkeep, self.learning_rate: lr_rate})
        return outputs

    def reward_weight(self, sess, x, generator):
        feed = {self.x: x, self.dropout_keep_prob: 1.0}
        snt_reward = sess.run(self.sent_reward, feed_dict=feed)
        feed = {generator.x: x}
        snt_log = sess.run(generator.sent_log, feed_dict=feed)
        weight = snt_reward - snt_log
        pos_w_batches = np.array([1.0 / (self.batch_size // 2)] * (self.batch_size // 2))
        neg_w_batches = weight[self.batch_size // 2 : ]
        return np.concatenate((pos_w_batches, neg_w_batches), axis=0)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    # def create_output_unit(self, params, midlayer):
    #     self.Wbo_list = []
    #     midlayer.insert(0, self.hidden_dim)
    #     midlayer.append(self.num_emb)
    #     assert len(midlayer) >= 2
    #     for i in xrange(1, len(midlayer)):
    #         print i
    #         self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[i - 1], midlayer[i]])))
    #         self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[i]])))
    #
    #     params.extend(self.Wbo_list)
    #
    #     def unit(hidden_memory_tuple):
    #         hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
    #         # hidden_state : batch x hidden_dim
    #         assert len(self.Wbo_list) == 2 * (len(midlayer) - 1)
    #         for j in range(len(self.Wbo_list) // 2 - 1):
    #             hidden_state = tf.nn.relu(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[2 * j], self.Wbo_list[2 * j + 1]))
    #         rewards = tf.nn.sigmoid(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[-2], self.Wbo_list[-1]))
    #         return rewards
    #
    #     return unit

    def create_output_unit(self, params, midlayer, num_highway=2):
        self.Wbo_list = []
        midlayer.insert(0, self.hidden_dim)
        midlayer.append(self.num_emb)
        assert len(midlayer) == 3
        self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[0], midlayer[1]])))
        self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1]])))
        for j in range(num_highway):
            self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1], midlayer[1]])))
            self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1]])))
            self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1], midlayer[1]])))
            self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1]])))
        self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[1], midlayer[2]])))
        self.Wbo_list.append(tf.Variable(self.init_matrix([midlayer[2]])))

        params.extend(self.Wbo_list)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            hidden_state = tf.nn.relu(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[0], self.Wbo_list[1]))
            # Use high-way network, FeedForward network also works
            for i in range(num_highway):
                tran = tf.nn.relu(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[2 + 4 * i], self.Wbo_list[3 + 4 * i]))
                gate = tf.nn.sigmoid(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[4 + 4 * i], self.Wbo_list[5 + 4 * i]))
                hidden_state = tran * gate + (1. - gate) * hidden_state
            # use sigmoid function to restrain the reward between 0 and 1
            rewards = tf.nn.sigmoid(tf.nn.xw_plus_b(hidden_state, self.Wbo_list[-2], self.Wbo_list[-1]))
            return rewards

        return unit

    def optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)
