import torch as t 
import utils 
import tqdm
import os
import ipdb 
import torch.nn.functional as F 
import torchnet as tnt 
import torchvision as tv 
from torch.utils import data 
from torch.autograd import Variable 
from transformer_net import TransformerNet
from config import opt 
from PackedVGG import VGG16




def train(**kwargs):

	for k_,v_ in kwargs.items():
		setattr(opt,k_,v_)

	if opt.vis is True:
		from visualize import Visualizer 
		vis = Visualizer(opt.env)

	transforms = tv.transforms.Compose([tv.transforms.Resize(opt.image_size),
									tv.transforms.CenterCrop(opt.image_size),
									tv.transforms.ToTensor(), #change value to (0,1)
									tv.transforms.Lambda( lambda x : x*255)]) #change value to (0,255)
	dataset = tv.datasets.ImageFolder(opt.data_root,transforms)

	dataloader = data.DataLoader(dataset,opt.batch_size)  #value is (0,255)

	transformer = TransformerNet()

	if opt.model_path:
		transformer.load_state_dict(t.load(opt.model_path,map_location=lambda _s,_: _s))

	vgg = VGG16().eval()
	for param in vgg.parameters():
		param.requires_grad = False 

	optimizer = t.optim.Adam(transformer.parameters(),opt.lr)

	style = utils.get_style_data(opt.style_path)
	vis.img('style',(style[0]*0.225 + 0.45).clamp(min=0,max=1))

	if opt.use_gpu:

		transformer.cuda()
		style = style.cuda()
		vgg.cuda()

	style_v = Variable(style.unsqueeze(0),volatile=True)
	features_style = vgg(style_v)
	gram_style     =[Variable(utils.gram_matrix(y.data)) for y in features_style]

	style_meter = tnt.meter.AverageValueMeter()
	content_meter = tnt.meter.AverageValueMeter()

	for epoch in range(opt.epoches):
		content_meter.reset()
		style_meter.reset()

		for ii,(x,_) in tqdm.tqdm(enumerate(dataloader)):

			optimizer.zero_grad()
			if opt.use_gpu:
				x = x.cuda() #(0,255)
			x = Variable(x)
			y = transformer(x) #(0,255)
			y = utils.normalize_batch(y) #(-2,2)
			x = utils.normalize_batch(x) #(-2,2)

			features_y = vgg(y)
			features_x = vgg(x)

		#calculate the content loss: it's only used relu2_2 
		# i think should add more layer's result to calculate the result like: w1*relu2_2+w2*relu3_2+w3*relu3_3+w4*relu4_3
			content_loss = opt.content_weight*F.mse_loss(features_y.relu2_2,features_x.relu2_2)
			content_meter.add(content_loss.data)

			style_loss = 0
			for ft_y,gm_s in zip(features_y,gram_style):

				gram_y = utils.gram_matrix(ft_y)
				style_loss += F.mse_loss(gram_y,gm_s.expand_as(gram_y))
			style_meter.add(style_loss.data)

			style_loss *=opt.style_weight

			total_loss = content_loss + style_loss 
			total_loss.backward()
			optimizer.step()

			if ( ii + 1 )% (opt.plot_every) == 0:

				if os.path.exists(opt.debug_file):
					ipdb.set_trace()

				vis.plot('content_loss',content_meter.value()[0])
				vis.plot('style_loss',style_meter.value()[0])

				vis.img('output',(y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0,max=1))
				vis.img('input',(x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0,max=1))

		vis.save([opt.env])
		t.save(transformer.state_dict(),'checkpoints/%s_style.pth'%epoch)



def stylize(**kwargs):

	'''
	generate the picture use the style of the style_picture.jpg
	'''
	for k_,v_, in kwargs.items():
		setattr(opt,k_,v_)

	content_image = tv.datasets.folder.default_loader(opt.content_path)
	
	content_transfrom = tv.transforms.Compose([
									tv.transforms.ToTensor(), #change value to (0,1)
									tv.transforms.Lambda( lambda x : x*255)]) #change value to (0,255)
	content_image = content_transfrom(content_image)
	content_image = Variable(content_image.unsqueeze(0),volatile=True)

	style_mode = TransformerNet().eval() # change to eval model 
	style_mode.load_state_dict(t.load(opt.model_path,map_location = lambda _s,_:_s))

	if opt.use_gpu == True:
		content_image = content_image.cuda()
		style_mode.cuda()

	output = style_mode(content_image)
	output_data = output.cpu().data[0]
	tv.utils.save_image((output_data / 255).clamp(min=0,max=1),opt.result_path)


if __name__ == '__main__':

	import fire 

	fire.Fire()