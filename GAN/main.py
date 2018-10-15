import os
import torch as t 
import ipdb
from torch.autograd import Variable 
from config import opt
from model import NetG,NetD
from torchnet import meter  
from tqdm import tqdm 
from torch.autograd import Variable 
import torchvision as tv  
from torchnet.meter import AverageValueMeter


def train(**kwargs):
	for k_,v_ in kwargs.items():
		setattr(opt,k_,v_)
    
	if opt.vis is True:
		from visualize import Visualizer
		vis = Visualizer(opt.env)
	transforms = tv.transforms.Compose([
				tv.transforms.Scale(opt.image_size),
				tv.transforms.CenterCrop(opt.image_size),
				tv.transforms.ToTensor(),
				tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

	dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)

	dataloader = t.utils.data.DataLoader(dataset,
									batch_size= opt.batch_size,
									shuffle = True,
									num_workers = opt.num_workers,
									drop_last = True)
	netg,netd = NetG(opt),NetD(opt)

	map_location = lambda storage, loc: storage
	if opt.netd_path:
		netd.load_state_dict(t.load(opt.netd_path,map_location = map_location))
	if opt.netg_path:
		netg.load_state_dict(t.load(opt.netg_path,map_location= map_location))

	optimizer_g = t.optim.Adam(netg.parameters(),opt.lr1,betas=(opt.beta1,0.999))
	optimizer_d = t.optim.Adam(netd.parameters(),opt.lr2,betas=(opt.beta1,0.999))

	criterion = t.nn.BCELoss()

	true_labels = Variable(t.ones(opt.batch_size))
	fake_labels = Variable(t.zeros(opt.batch_size))

	fix_noises = Variable(t.randn(opt.batch_size,opt.nz,1,1))

	noises = Variable(t.randn(opt.batch_size,opt.nz,1,1))

	if opt.use_gpu:
		netd.cuda()
		netg.cuda()
		criterion.cuda()
		true_labels,fake_labels = true_labels.cuda(),fake_labels.cuda()
		fix_noises,noises = fix_noises.cuda(),noises.cuda()

	errord_meter = AverageValueMeter()
	errorg_meter = AverageValueMeter()

	epoches =range(opt.max_epoch)
	for epoch in iter(epoches):

		for ii, (img,_) in tqdm(enumerate(dataloader)):

			real_img = Variable(img)
			if opt.use_gpu:
				real_img = real_img.cuda()

			if (ii +1 )% opt.d_every == 0:

				optimizer_d.zero_grad()
				output = netd(real_img)
				error_d_real = criterion(output,true_labels)
				error_d_real.backward()

				noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
				fake_img = netg(noises).detach()
				fake_output = netd(fake_img)
				error_d_fake = criterion(fake_output,fake_labels)
				error_d_fake.backward()

				optimizer_d.step()

				error_d =error_d_real + error_d_fake 
				errord_meter.add(error_d.data)


			if (ii + 1)%opt.g_every == 0:

				optimizer_g.zero_grad()
				noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
				fake_img = netg(noises)
				fake_output = netd(fake_img)

				error_g = criterion(fake_output,true_labels)
				error_g.backward()
				optimizer_g.step()

				errorg_meter.add(error_g.data)


			if opt.vis and ii%opt.plot_every == opt.plot_every -1 :
				#visualize

				if os.path.exists(opt.debug_file):
					ipdb.set_trace()

				fix_fake_img = netg(fix_noises)
				vis.images(fix_fake_img.data.cpu().numpy()[:16]*0.5 + 0.5 ,win='fixfake')
				vis.images(real_img.data.cpu().numpy()[:16]*0.5+0.5,win='real')

				vis.plot('errord',errord_meter.value()[0])
				vis.plot('errorg',errorg_meter.value()[0])

			if (epoch+1) % opt.save_every ==0:

				tv.utils.save_image(fix_fake_img.data[:64],'%s/%s.png'%(opt.save_path,epoch),normalize=True,range=(-1,1))
				t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
				t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
				errord_meter.reset()
				errorg_meter.reset()

def generate(**kwargs):
	'''
	random create caton images and chose the highest scords top 60
	'''
	for k_,v_, in kwargs.items():
		setattr(opt,k_,v_)


	netg,netd = NetG(opt).eval(),NetD(opt).eval()
	noises = Variable(t.randn(opt.gen_search_num,opt.nz,1,1).normal_(opt.gen_mean,opt.gen_std))
	

	map_location = lambda storage,loc:storage 
	netd.load_state_dict(t.load(opt.netd_path,map_location=map_location))
	netg.load_state_dict(t.load(opt.netg_path,map_location=map_location))

	if opt.use_gpu is True:
		noises.cuda()
		netd.cuda()
		netg.cuda()
		ipdb.set_trace()

	fake_img = netg(noises)
	scores = netd(fake_img).data

	indexs = scores.topk(opt.gen_num)[1]
	result = []

	for ii in indexs:
		result.append(fake_img.data[ii])

	tv.utils.save_image(t.stack(result),opt.gen_img,normalize=True,range=(-1,1))


if __name__ == '__main__':
	import fire
	fire.Fire()
























