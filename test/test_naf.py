import gym
import logging
from agents.NAF import NAF
import numpy as np
import tensorflow as tf
import time
import pyprind
from normalize_env import Normalization
import ConfigParser
import argparse

def read_configuration(file):
    configParser = ConfigParser.SafeConfigParser()   
    configParser.read(file)

    config = {}

    config['show_parameters'] = configParser.get('debug','show_parameters')
    config['statistic'] = configParser.getboolean('debug', 'statistic')


    # get the configuration for gym
    config['env'] = configParser.get('gym','env')
    config['monitor'] = configParser.getboolean('gym','monitor')
    config['monitor_dir'] = configParser.get('gym','monitor_dir')
    config['video'] = configParser.getboolean('gym','video')
    config['max_steps'] = configParser.getint('gym','max_steps')
    config['max_episodes'] = configParser.getint('gym','max_episodes')


    # get the configuration for gpu
    config['gpu'] = configParser.getboolean('tensorflow','gpu')
    config['show_device_info'] = configParser.getboolean('tensorflow','show_device_info')


    # get the configuration for agent
    config['update_per_iteration'] = configParser.getint('agent','update_per_iteration')
    config['mini_batch_size'] = configParser.getint('agent','mini_batch_size')
    config['max_buffer_size'] = configParser.getint('agent','max_buffer_size')
    config['discount'] = configParser.getfloat('agent','discount')
    config['batch_norm'] = configParser.getboolean('agent','batch_norm')
    config['lr'] = configParser.getfloat('agent','lr')
    config['soft_lr'] = configParser.getfloat('agent','soft_lr')

    layers = configParser.get('agent','hidden_layers')
    str_list = layers.split(',')
    config['hidden_layers'] = []
    for string in str_list:
        config['hidden_layers'].append(int(string))


    return config



def configurate_tf(config):
    tf_config = None

    if config['gpu']:
        device = {'GPU': 1}
    else:
        device = {'GPU': 0}

    tf.logging.set_verbosity(tf.logging.ERROR)

    tf_config = tf.ConfigProto(device_count=device, log_device_placement=config['show_device_info'])

    return tf_config



def show_parameters(config):
    print '==================== parameters ==================='
    print 'max_episodes = {}'.format(config['max_episodes'])
    print 'max_steps = {}'.format(config['max_steps'])
    print 'update_per_iteration = {}'.format(config['update_per_iteration'])
    print 'mini_batch_size = {}'.format(config['mini_batch_size'])
    print 'max_buffer_size = {}'.format(config['max_buffer_size'])
    print 'discount = {}'.format(config['discount'])
    print 'batch_norm = {}'.format(config['batch_norm'])
    print 'lr = {}'.format(config['lr'])
    print 'soft_lr = {}'.format(config['soft_lr'])
    print 'hidden_layers = {}'.format(config['hidden_layers'])
    print '==================== parameters ==================='







def main(_):
    
    # get the configuration from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="indicates configuration file")
    args = parser.parse_args()

    config_file = 'naf.cfg' if args.f is None else args.f


    # we first hardcode the file path here
    config = read_configuration(config_file)

    tf_config = configurate_tf(config)

    if config['show_parameters']:
        show_parameters(config)

    with tf.Session(config=tf_config) as sess:

        # start the environment
        env = Normalization(gym.make(config['env']))

        if config['monitor']:
            if config['video']:
                env.monitor.start(config['monitor_dir'],force=True)
            else:
                env.monitor.start(config['monitor_dir'],force=True,video_callable=lambda count: False)

        # NOTE: still not sure what's the meaning of shape
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]


        # create the agent
        agent = NAF(sess, env, state_dim, action_dim, 
            mini_batch_size=config['mini_batch_size'], max_buffer_size=config['max_buffer_size'], 
            update_per_iteration=config['update_per_iteration'], discount=config['discount'], batch_norm=config['batch_norm'],
            learning_rate=config['lr'],tau=config['soft_lr'],hidden_layers=config['hidden_layers'])

        start_time = time.time()

        for i in xrange(config['max_episodes']):

            # first reset the environment
            s = env.reset()    

            agent.reset_exploration()

            for j in pyprind.prog_bar(xrange(config['max_steps'])):

                if config['video']:
                    env.render()


                action = agent.get_action(s)

                sprime, r, terminal, info = env.step(action)

                agent.learn(s, action, sprime, r, terminal)

                s = sprime

                if terminal:
                    # if we reach the terminal state, we restart the game
                    # however, before that, we may want to do something for to show the statistic
                    break

        elapsed_time = time.time() - start_time

        print 'elapsed time is ' + str(elapsed_time)

        if config['monitor']:
            env.monitor.close()






if __name__ == '__main__':
    tf.app.run()