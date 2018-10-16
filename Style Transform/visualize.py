from itertools import chain
import visdom
import torch
import time
import torchvision as tv 
import numpy as np 

class Visualizer():
	'''
	boxes the prime step of Visdom ,but you still can use it by 'self.vis.function'
	'''
	def __init__(self,env='default',**kwargs):
		self.vis = visdom.Visdom(env=env,use_incoming_socket=False,**kwargs)

		self.index = {} # save the index of the point ex:('loss',23)
		self.log_text = ''

	def reinit(self,env='default',**kwargs):
		'''
		modify the config of the visdom
		'''
		self.vis = visdom.Visdom(env = env,use_incoming_socket=False,**kwargs)
		return self

	def plot_many(self,d):

		'''
		plot several figure once 
		@params d: dict (name,value) i.e. ('loss',0.11)
		'''
		for k,v in d.items():
			self.plot(k,v)
	def img_many(self,d):
		for k,v in d.items():
			self.img(k,v)

	def plot(self,name,y):
		'''
		self.plot('loss',1.00)
		'''
		x = self.index.get(name,0) #get(name) name=index.key ,defalt_value=0
		self.vis.line(Y=np.array([y]),X=np.array([x]),win=(name),opts=dict(title=name),update=None \
			if x==0 else 'append')
		self.index[name] = x+1

	def img(self,name,img_):
		'''
		self.img('input_img',t.Tensor(64,64))
		'''
		if len(img_.size()) < 3:
			img_ = img_.cpu().unsqueeze(0)
		self.vis.image(img_.cpu(),win=(name),opts=dict(title=name))

	def img_grid(self,name,input_3d):
		'''
		a batch_size's picture will transform to a grid picture i.e (36,64,64)
		will change to 6*6 grid ,each picture's size is 64*64
		'''
		self.img(name,tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1,min=0)))

	def log(self, info, win='log_text'):
		"""
		self.log({'loss':1,'lr':0.0001})
		"""
		self.log_text += ('[{time}] {info} <br>'.format(
			time=time.strftime('%m%d_%H%M%S'),
			info=info))
		self.vis.text(self.log_text, win=win)

	def __getattr__(self, name):
		return getattr(self.vis, name)