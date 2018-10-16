class Config(object):

	image_size = 256
	batch_size = 8
	data_root  = 'data/'
	num_workers =4
	vis = True
	use_gpu = True
	style_path ='style.jpg'
	lr = 1e-3
	env = 'neural-style'
	plot_every = 10 
	epoches =2 
	content_weight = 1e5 
	style_weight   = 1e10 

	model_path = None 
	debug_file = '/tmp/debugnn'

	content_path = 'input.png'
	result_path  = 'output.png'


opt = Config()
