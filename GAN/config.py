class Config(object):

	data_path = 'data/'
	num_workers = 4
	image_size = 51#96
	batch_size = 16
	max_epoch  =200
	lr1 = 2e-4 #0.0002
	lr2 = 2e-4 
	beta1 = 0.5
	use_gpu = True
	nz = 100 #the axis of noise
	ngf = 64
	ndf = 64 

	save_path='imgs/'

	vis = True
	env = 'GAN'

	plot_every = 20 
	debug_file = '/tmp/debuggan'
	d_every =  1
	g_every =  1
	decay_every = 10
	save_every  =10
	netd_path  = None
	netg_path  = None 

	gen_img    = 'result.png'
	gen_num    =  16 #64
	gen_search_num = 64#512
	gen_mean = 0
	gen_std  = 1

opt = Config()