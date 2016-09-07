from logging import getLogger
from replay_buffer import ReplayBuffer
from OUProcess import OUProcess
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *



logger = getLogger(__name__)




class DDPG(object):
    def __init__(self, sess, env, state_dim, action_dim, max_buffer_size=100000, update_per_iteration=5, mini_batch_size=64,
        discount=0.99, batch_norm=True, actor_learning_rate=0.0001, critic_learning_rate=0.001, tau=0.001,
        hidden_layers=[400,300]):


        self.session = sess
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lb = self.env.action_space.low
        self.action_ub = self.env.action_space.high
        self.discount = discount
        self.batch_norm = batch_norm
        self.mini_batch_size = mini_batch_size
        self.update_per_iteration = update_per_iteration
        self.hidden_layers = hidden_layers

        self.replay_buffer = ReplayBuffer(max_buffer_size, state_dim, action_dim)
        self.exploration = OUProcess(self.action_dim)


        # we define the operations that is used in this algorithms
        self.critic = {}
        self.critic['x'], self.critic['u'], self.critic['is_train'], self.critic['q'], self.critic['variables'] = self.create_critic_network(is_target=False)
        
        self.target_critic = {}
        self.target_critic['x'], self.target_critic['u'], _, self.target_critic['q'], self.target_critic['variables'] = self.create_critic_network(is_target=True)

        self.actor = {}
        self.actor['x'], self.actor['is_train'], self.actor['a'], self.actor['variables'] = self.create_actor_network(is_target=False)

        self.target_actor = {}
        self.target_actor['x'], _, self.target_actor['a'], self.target_actor['variables'] = self.create_actor_network(is_target=True)





        self.critic_optimization = {}
        with tf.name_scope('critic_optimization'):
            self.critic_optimization['y'] = tf.placeholder(tf.float32, shape=(None, 1), name='y')
            self.critic_optimization['loss'] = tf.reduce_mean(tf.squared_difference(self.critic['q'], self.critic_optimization['y']), name='loss')
            self.critic_optimization['optimize'] = tf.train.AdamOptimizer(critic_learning_rate).minimize(self.critic_optimization['loss'])



        # define operation to get y
        self.y_compute = {}
        with tf.name_scope('y'):
            # y = reward + (1-terminal) * gamma * target_q
            self.y_compute['r'] = tf.placeholder(tf.float32, shape=(None, 1))
            self.y_compute['t'] = tf.placeholder(tf.int8, shape=(None, 1))
            self.y_compute['q'] = tf.placeholder(tf.float32, shape=(None, 1))
            temp = tf.to_float(self.y_compute['t'])
            temp = tf.mul(temp, -1.0)
            temp = tf.add(temp, 1.0)
            self.y_compute['y'] = tf.add(self.y_compute['r'], tf.mul(tf.mul(self.y_compute['q'], self.discount), temp))

        # define the operation to get the gradient of Q with respect to action
        self.action_gradients = {}
        with tf.name_scope('action_grads'):
            self.action_gradients["action_grads"] = tf.gradients(self.critic['q'], self.critic['u'])


        self.actor_optimization = {}
        with tf.name_scope('actor_optimization'):
            # first define the placeholder for the gradient of Q with respect to action
            self.actor_optimization['action_grads'] = tf.placeholder(tf.float32, shape=(None, self.action_dim))
            # since actor are using gradient ascent, we add the minus sign
            self.actor_optimization['actor_variable_grads'] = tf.gradients(self.actor['a'], self.actor['variables'], -self.actor_optimization['action_grads'])
            self.actor_optimization['optimize'] = tf.train.AdamOptimizer(actor_learning_rate).apply_gradients(zip(self.actor_optimization['actor_variable_grads'], self.actor['variables']))

        self.soft_update_list = []
        with tf.name_scope("soft_update"):
            for source, dest in zip(self.critic['variables'], self.target_critic['variables']):
                self.soft_update_list.append(dest.assign(tf.mul(source, tau) + tf.mul(dest, 1.0-tau)))
            for source, dest in zip(self.actor['variables'], self.target_actor['variables']):
                self.soft_update_list.append(dest.assign(tf.mul(source, tau) + tf.mul(dest, 1.0-tau)))

        # after define the computation, we initialize all the varialbes
        self.session.run(tf.initialize_all_variables())

        summary_writer = tf.train.SummaryWriter('critic.graph', graph_def=self.session.graph)





    def create_actor_network(self, is_target):

        scope = 'tar_actor' if is_target else 'actor'

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='observation')

            # this is used for determine which mode, training or evalutation, for batch normalization
            if self.batch_norm:
                # for target network, is alway evaluation mode
                is_train = False if is_target else tf.placeholder(tf.bool, name='is_train')
            else:
                is_train = None


            net = x

            for hidden_unit_num in self.hidden_layers:
                if self.batch_norm:
                    net = fully_connected(inputs=net, activation_fn=None, num_outputs=hidden_unit_num)
                    # NOTE : we set the updates_collections to None to force the updates of mean and variance in place
                    net = batch_norm(inputs=net, center=True, scale=True, activation_fn=tf.nn.relu, is_training = is_train, updates_collections=None)
                else:
                    net = fully_connected(inputs=net, activation_fn=tf.nn.relu, num_outputs=hidden_unit_num)

            net = fully_connected(inputs=net, activation_fn=tf.tanh, num_outputs=self.action_dim, 
                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                    biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        
        # get all the trainable variable from this scope
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return x, is_train, net, variables


    def create_critic_network(self, is_target):
        scope = 'tar_critic' if is_target else 'critic'

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='observation')
            u = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='actions')


            # this is used for determine which mode, training or evalutation, for batch normalization
            if self.batch_norm:
                # for target network, is alway evaluation mode
                is_train = False if is_target else tf.placeholder(tf.bool, name='is_train')
            else:
                is_train = None

            # first concatenate the input
            # NOTE : this is different architecture from the original paper, we include the action from the first layer
            with tf.name_scope('merge'):
                net = tf.concat(1,[x, u])

            for hidden_unit_num in self.hidden_layers:
                if self.batch_norm:
                    net = fully_connected(inputs=net, activation_fn=None, num_outputs=hidden_unit_num)
                    # NOTE : we set the updates_collections to None to force the updates of mean and variance in place
                    net = batch_norm(inputs=net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_train, updates_collections=None)
                else:
                    net = fully_connected(inputs=net, activation_fn=tf.nn.relu, num_outputs=hidden_unit_num)

            net = fully_connected(inputs=net, activation_fn=None, num_outputs=1,
                weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # get all the trainable variable from this scope
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return x, u, is_train, net, variables



    # define the functions for executing operations
    def predict_target_q(self, x, u):
        return self.session.run(self.target_critic['q'], feed_dict={
            self.target_critic['x']:x, 
            self.target_critic['u']:u
            }) 


    def predict_target_action(self, x):
        return self.session.run(self.target_actor['a'], feed_dict={
            self.target_actor['x']:x
            })

    def get_y(self, q, r, t):
        return self.session.run(self.y_compute['y'], feed_dict={
            self.y_compute['r']:r,
            self.y_compute['q']:q,
            self.y_compute['t']:t
            })

    def optimize_critic(self, x, u, is_train, y):
        if self.batch_norm:
            return self.session.run(self.critic_optimization['optimize'], feed_dict={
                self.critic['x']:x,
                self.critic['u']:u,
                self.critic['is_train']:is_train,
                self.critic_optimization['y']:y
                })
        else:
            return self.session.run(self.critic_optimization['optimize'], feed_dict={
                self.critic['x']:x,
                self.critic['u']:u,
                self.critic_optimization['y']:y
                })


    def predict_action(self, x, is_train):
        if self.batch_norm:
            return self.session.run(self.actor['a'], feed_dict={
                self.actor['x']:x, 
                self.actor['is_train']:is_train
                })
        else:
            return self.session.run(self.actor['a'], feed_dict={
                self.actor['x']:x
                })

    def action_grads(self, x, u, is_train):
        if self.batch_norm:
            return self.session.run(self.action_gradients["action_grads"], feed_dict={
                self.critic['x']:x,
                self.critic['u']:u,
                self.critic['is_train']:is_train
                })
        else:
            return self.session.run(self.action_gradients["action_grads"], feed_dict={
                self.critic['x']:x,
                self.critic['u']:u
                })

                
    def optimize_actor(self, x, a_grads, is_train):
        if self.batch_norm:
            return self.session.run(self.actor_optimization['optimize'], feed_dict={
                self.actor['x']:x,
                self.actor['is_train']:is_train,
                self.actor_optimization['action_grads']:a_grads
                })
        else:
            return self.session.run(self.actor_optimization['optimize'], feed_dict={
                self.actor['x']:x,
                self.actor_optimization['action_grads']:a_grads
                })


                
    def soft_update(self):
        self.session.run(self.soft_update_list)

    def get_action(self, s):

        # first make sure the s have the valid form
        s = np.reshape(s, (1, self.state_dim))

        a = self.predict_action(s, False)

        # a is a list with mini_batch size of 1, so we need the first element of is_train
        return self.exploration.add_noise(a[0], self.action_lb, self.action_ub)   



    def learn(self, s, a, sprime, r, t):

        # first add the sample to the replay buffer
        self.replay_buffer.add(s, a, sprime, r, t)


        # we start learning if we have enough sample for one minibatch
        if self.replay_buffer.get_size() > self.mini_batch_size:
            
            # we do the update with several batch in each turn
            for i in xrange(self.update_per_iteration):
                state_set, action_set, sprime_set, reward_set, terminal_set = self.replay_buffer.sample_batch(self.mini_batch_size)


                # first optimize the critic
                # compute Q'
                q = self.predict_target_q(sprime_set, self.predict_target_action(sprime_set))

                # compute y = r + gamma * Q'
                y = self.get_y(q, reward_set, terminal_set)

                # optimize critic using y, and batch normalization
                self.optimize_critic(state_set, action_set, True, y)


                # then optimize the actor
                actions = self.predict_action(state_set, True)
                a_grads = self.action_grads(state_set, actions, False)
                # NOTE: the tf.gradient return a list of len(actions), so we need to take the first element from it
                self.optimize_actor(state_set, a_grads[0], True)


                # using soft update to update target networks
                self.soft_update()


    def reset_exploration(self):
        self.exploration.reset()




        




