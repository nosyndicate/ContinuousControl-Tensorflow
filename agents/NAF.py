from logging import getLogger
from replay_buffer import ReplayBuffer
from OUProcess import OUProcess
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *



logger = getLogger(__name__)




class NAF(object):
    def __init__(self, sess, env, state_dim, action_dim, max_buffer_size=100000, update_per_iteration=5, mini_batch_size=64,
        discount=0.99, batch_norm=True, learning_rate=1e-3, tau=0.001, hidden_layers=[200,200]):


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


    
        self.network = {}
        self.network['x'], self.network['u'], self.network['is_train'], self.network['V'], self.network['P'], \
            self.network['M'], self.network['Q'], self.network['variables'] = self.create_networks(is_target=False)
        
        self.target = {}
        self.target['x'], self.target['u'], _, self.target['V'], self.target['P'], \
            self.target['M'], self.target['Q'], self.target['variables'] = self.create_networks(is_target=True)

        #define optimization operations
        self.network_optimization = {}
        with tf.name_scope('optimization'):
            self.network_optimization['y'] = tf.placeholder(tf.float32, shape=(None, 1), name='y')
            self.network_optimization['loss'] = tf.reduce_mean(tf.squared_difference(self.network['Q'], self.network_optimization['y']), name='loss')
            self.network_optimization['optimize'] = tf.train.AdamOptimizer(learning_rate).minimize(self.network_optimization['loss'])


        #define the operations for compute y value
        self.y_compute = {}
        with tf.name_scope('y'):
            # y = reward +  (1-terminal) * gamma * V
            self.y_compute['r'] = tf.placeholder(tf.float32, shape=(None, 1))
            self.y_compute['t'] = tf.placeholder(tf.int8, shape=(None, 1))
            self.y_compute['v'] = tf.placeholder(tf.float32, shape=(None, 1))
            self.y_compute['y'] = tf.to_float(self.y_compute['t'])
            self.y_compute['y'] = tf.mul(self.y_compute['y'], -1.0)
            self.y_compute['y'] = tf.add(self.y_compute['y'], 1.0)
            self.y_compute['y'] = tf.add(self.y_compute['r'], tf.mul(tf.mul(self.y_compute['v'], self.discount), self.y_compute['y']))


        # define the soft update operation between the normal networks and target networks
        self.soft_update_list = []
        with tf.name_scope('soft_update'):
            for source, dest in zip(self.network['variables'], self.target['variables']):
                self.soft_update_list.append(dest.assign(tf.mul(source, tau) + tf.mul(dest, 1.0-tau)))


        # after define the computation, we initialize all the varialbes
        self.session.run(tf.initialize_all_variables())

        summary_writer = tf.train.SummaryWriter('naf.graph', graph_def=self.session.graph)


    def create_networks(self, is_target):

        scope = 'tar_naf' if is_target else 'naf'

        with tf.variable_scope(scope):
            x = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='observation')
            u = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='actions')


            # this is used for determine which mode, training or evalutation, for batch normalization
            if self.batch_norm:
                # for target network, is alway evaluation mode
                is_train = False if is_target else tf.placeholder(tf.bool, name='is_train')
            else:
                is_train = None

            # define operations for the value function
            with tf.variable_scope('V'):
                V = x
                # add in the hidden layers
                for hidden_unit_num in self.hidden_layers:
                    if self.batch_norm:
                        V = fully_connected(inputs=V, activation_fn=None, num_outputs=hidden_unit_num)
                        # NOTE : we set the updates_collections to None to force the updates of mean and variance in place
                        V = batch_norm(inputs=V, center=True, scale=True, activation_fn=tf.nn.relu, is_training = is_train, updates_collections=None)
                    else:
                        V = fully_connected(inputs=V, activation_fn=tf.nn.relu, num_outputs=hidden_unit_num)
                
                # add in the last layer
                V = fully_connected(inputs=V,activation_fn=None,num_outputs=1)

            # define operations for compute covariance matrix
            with tf.variable_scope('L'):
                L = x
                # add in the hidden layers
                for hidden_unit_num in self.hidden_layers:
                    if self.batch_norm:
                        L = fully_connected(inputs=L, activation_fn=None, num_outputs=hidden_unit_num)
                        # NOTE : we set the updates_collections to None to force the updates of mean and variance in place
                        L = batch_norm(inputs=L, center=True, scale=True, activation_fn=tf.nn.relu, is_training = is_train, updates_collections=None)
                    else:
                        L = fully_connected(inputs=L, activation_fn=tf.nn.relu, num_outputs=hidden_unit_num)

                
                L = fully_connected(inputs=L,activation_fn=None, num_outputs=(self.action_dim*(self.action_dim+1)/2))

                #construct upper triangular matrix U
                pivot = 0
                rows = []
                for index in xrange(self.action_dim):
                    count = self.action_dim - index

                    # slice one element at point pivot from the second dimension and apply exp to it 
                    # NOTE, first dimension indicate the batch, -1 means all element in this dimension are in slice
                    diag_elem = tf.exp(tf.slice(L, (0, pivot), (-1, 1)))

                    # slice the next count - 1 element from the second dimension
                    # count is the number of non-zero element in each row
                    # NOTE: index getting bigger, so count get smaller
                    non_diag_elems = tf.slice(L, (0, pivot+1), (-1, count-1))

                    # concate the tensor to form one row of the matrix 
                    non_zero_elements = tf.concat(1, (diag_elem, non_diag_elems))

                    # ((0, 0), (index, 0)) is the paddings
                    # since we have two-d matrix, so the tuple has two elements
                    # for the first (0,0), specify the first dimension
                    # the first 0 means padding nothing, the second 0 means padding before the elements (-1 means after)
                    # (index, 0) specify the padding for second dimension, which is what we want
                    # (index, 0) mean padding index number before the elements
                    row = tf.pad(non_zero_elements, ((0, 0), (index, 0)))
                    rows.append(row)

                    # take off the elements we already used
                    pivot += count


                # Packs a list of rank-R tensors into one rank-(R+1) tensor.
                # axis = 1 mean the second dimensions
                # NOTE : this will get upper triangular matrix U not L
                L = tf.pack(rows, axis=1)

                # convariance matrix P = L*L^{T} = U^{T}*U
                P = tf.batch_matmul(tf.transpose(L,perm=[0,2,1]), L)

            # define operations for compute Mu
            with tf.variable_scope('M'):
                M = x
                # add in the hidden layers
                for hidden_unit_num in self.hidden_layers:
                    if self.batch_norm:
                        M = fully_connected(inputs=M, activation_fn=None, num_outputs=hidden_unit_num)
                        # NOTE : we set the updates_collections to None to force the updates of mean and variance in place
                        # see https://github.com/tensorflow/tensorflow/issues/1122
                        M = batch_norm(inputs=M, center=True, scale=True, activation_fn=tf.nn.relu, is_training = is_train, updates_collections=None)
                    else:
                        M = fully_connected(inputs=M, activation_fn=tf.nn.relu, num_outputs=hidden_unit_num)
                        
                # add in the last layer
                M = fully_connected(inputs=M,activation_fn=tf.tanh,num_outputs=self.action_dim)

            #define operations for compute Advantage function
            with tf.name_scope('A'):
                # first expand the u-M to a 2-d tensor for multiplication
                # NOTE: it's actually a 3-d tensor, but we ignore the first dim which is the batch
                # u is two-d matrix, first dimension is the batch
                # so u is actually a row vector after expand_dim
                Aprime = tf.expand_dims(u - M, -1)
                # fix the dimension for batch, transpose each instance
                A = tf.transpose(Aprime, perm=[0,2,1])
                # A = -1/2 * (u-M)^{T} * P * (u-M)
                A = -tf.batch_matmul(tf.batch_matmul(Aprime, P), A)/2
                # make sure the shape is batch_size *  1 for A, -1 mean that dim is automatically computed
                # after last step, each A is now a 1*1 matrix, we reshape it to get scalar
                A = tf.reshape(A,[-1,1])

            with tf.name_scope('Q'):
                 Q = A + V



        # get all the trainable variable from this scope
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        #return x, u, is_train, V, P, M, Q, variables
        return x, u, is_train, V, P, M, Q, variables

    def predict_target_v(self, x):
        return self.session.run(self.target['V'], feed_dict={
            self.target['x']:x
            })

    def get_y(self, v, r, t):
        return self.session.run(self.y_compute['y'], feed_dict={
            self.y_compute['r']:r,
            self.y_compute['v']:v,
            self.y_compute['t']:t
            })

    def optimize_network(self, x, u, is_train, y):
        if self.batch_norm:
            feed_dict = { self.network['x']:x, self.network['u']:u, self.network['is_train']:is_train, self.network_optimization['y']:y }
        else:
            feed_dict = { self.network['x']:x, self.network['u']:u, self.network_optimization['y']:y}

        return self.session.run(self.network_optimization['optimize'], feed_dict=feed_dict)


    def predict_action(self, x, is_train):
        if self.batch_norm:
            feed_dict = { self.network['x']:x, self.network['is_train']:is_train }
        else:
            feed_dict = { self.network['x']:x }
        return self.session.run([self.network['M'], self.network['P']], feed_dict=feed_dict)


    def get_action(self, s):
        
        s = np.reshape(s, (1, self.state_dim))    

        a, covariance = self.predict_action(s, False)


        return self.exploration.add_noise(a[0], self.action_lb, self.action_ub)   


    def soft_update(self):
        self.session.run(self.soft_update_list)




    def learn(self, s, a, sprime, r, terminal):
        # first add the sample to the replay buffer
        self.replay_buffer.add(s, a, sprime, r, terminal)

        # we start learning if we have enough sample for one minibatch
        if self.replay_buffer.get_size() > self.mini_batch_size:
            
            # we do the update with several batch in each turn
            for i in xrange(self.update_per_iteration):
                state_set, action_set, sprime_set, reward_set, terminal_set = self.replay_buffer.sample_batch(self.mini_batch_size)

                # compute V'
                v = self.predict_target_v(sprime_set)

                # compute y = r + gamma * V'
                y = self.get_y(v, reward_set, terminal_set)

                # optimize critic using y, and batch normalization
                self.optimize_network(state_set, action_set, True, y)

                # using soft update to update target networks
                self.soft_update()



    
    def reset_exploration(self):
        self.exploration.reset()




