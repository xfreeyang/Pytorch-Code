import torchvision as tv 
import torch as t 
IMAGENET_MEAN =  [0.485,0.456,0.406]
IMAGENET_STD  =  [0.229,0.224,0.225]

def gram_matrix(y):
	'''
	conclulate the gram matrix of the feature map 
	input shape : b,c,h,w 
	output shape : b,c,c 
	'''
	(b,ch,h,w) = y.size()
	features = y.view(b,ch,w*h)
	features_t = features.transpose(1,2)
	gram = features.bmm(features_t)/(ch*h*w)

	return gram 



def get_style_data(path):

	'''
	load style_picture 
	input :  picture path 
	output :  1*c*h*w, about(-2,2) tensor
	'''
	style_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
											tv.transforms.Normalize(mean= IMAGENET_MEAN,std = IMAGENET_STD),])
	style_image = tv.datasets.folder.default_loader(path) #read_image(0,255)
	style_tensor = style_transform(style_image)#Totensor(0,1) Normalize(-2.1,2.7)
	return style_tensor.squeeze(0)

def normalize_batch(batch):

	'''
	input: b,ch,h,w  value about 0-255 is a Variable
	output:b,ch,h,w, value about -2~2  is a Variable 
	'''

	mean = batch.data.new(IMAGENET_MEAN).view(1,-1,1,1)
	std  = batch.data.new(IMAGENET_STD).view(1,-1,1,1)
	mean = t.autograd.Variable(mean.expand_as(batch.data))
	std  = t.autograd.Variable(std.expand_as(batch.data))

	return (batch/255.0 - mean) /std 
