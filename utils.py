import ConfigParser
import tensorflow as tf


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



	config['gpu'] = configParser.getboolean('tensorflow','gpu')
	config['show_device_info'] = configParser.getboolean('tensorflow','show_device_info')




	config['agent'] = configParser.get('agent','agent')

	config['update_per_iteration'] = configParser.getint('agent','update_per_iteration')
	config['mini_batch_size'] = configParser.getint('agent','mini_batch_size')
	config['max_buffer_size'] = configParser.getint('agent','max_buffer_size')
	config['discount'] = configParser.getfloat('agent','discount')
	config['batch_norm'] = configParser.getboolean('agent','batch_norm')
	
	if (config['agent']=='ddpg'):
		config['actor_lr'] = configParser.getfloat('agent','actor_lr')
		config['critic_lr'] = configParser.getfloat('agent','critic_lr')
	elif (config['agent']=='naf'):
		config['lr'] = configParser.getfloat('agent','lr')
	elif (config['agent']=='mnaf'):
		config['actor_lr'] = configParser.getfloat('agent','actor_lr')
		config['critic_lr'] = configParser.getfloat('agent','critic_lr')
		config['component_number'] = configParser.getint('agent','component_number')

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
	if (config['agent']=='ddpg'):
		print 'actor_lr = {}'.format(config['actor_lr'])
		print 'critic_lr = {}'.format(config['critic_lr'])
	elif (config['agent']=='naf'):
		print 'lr = {}'.format(config['lr'])
	elif (config['agent']=='mnaf'):
		print 'actor_lr = {}'.format(config['actor_lr'])
		print 'critic_lr = {}'.format(config['critic_lr'])
		print 'component_number = {}'.format(config['component_number'])
	print 'soft_lr = {}'.format(config['soft_lr'])
	print 'hidden_layers = {}'.format(config['hidden_layers'])
	print '==================== parameters ==================='





