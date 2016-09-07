import gym
import logging
import utils
import NAF
import numpy as np
import tensorflow as tf
import time
import pyprind
from normalize_env import Normalization





def main(_):
    # we first hardcode the file path here
    config = utils.read_configuration('naf.conf')

    tf_config = utils.configurate_tf(config)

    if config['show_parameters']:
        utils.show_parameters(config)

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
        agent = NAF.NAF(sess, env, state_dim, action_dim, 
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