# author: Wu
# Created: 2023/8/15
# baseline of model

import shutil
import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataset
import model
from torch.utils.tensorboard import SummaryWriter

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


def train(config):
	# tensorboard
	if os.path.exists(os.path.join(config.snapshots_folder, 'tb')):
		shutil.rmtree(os.path.join(config.snapshots_folder, 'tb'))
	if not os.path.exists(os.path.join(config.snapshots_folder, 'sample')):
		os.makedirs(os.path.join(config.snapshots_folder, 'sample'))
	os.makedirs(os.path.join(config.snapshots_folder, 'tb'))
	writer = SummaryWriter(log_dir=os.path.join(config.snapshots_folder, 'tb'))
	global_step = 0
	# load model
	os.environ['CUDA_VISIBLE_DEVICES']=config.device
	if config.model == 'ZeroDCE':
		net = model.ZeroDCE().cuda()
	elif config.model == 'ECNet': 
		net = model.ECNet(config).cuda()
	else:
		print(f'model {config.model}  is not available') 
		exit()
	net.apply(weights_init)
	if config.load_pretrain == True:
		net.load_state_dict(torch.load(config.pretrain_dir))
	if len(config.device)>1:
		net = nn.DataParallel(net)
	# build dataset
	train_dataset = dataset.LOL_dataset(config.LI_path, config.HI_path)	if config.dataset=='lol' else dataset.EE_dataset(config.LI_path, config.HI_path)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	# optimizer
	if config.adam:
		optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # adjust beta1 to momentum
	else:
		optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)  
	# train
	net.train()
	for epoch in range(config.num_epochs):
		for iteration, data in enumerate(train_loader):
			LI, HI = data
			LI = LI.cuda()
			HI = HI.cuda()
			loss, loss_h, loss_rec, loss_z, loss_color = net.loss_stage_1(LI, HI, epoch >= config.l_rec_epoch)
			# optimize
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm(net.parameters(),config.grad_clip_norm)
			optimizer.step()
			# log
			if ((iteration+1) % config.log_iter) == 0:
				print(f"epoch: {epoch+1}, iter: {iteration+1}, loss: {loss.item()}, loss_h: {loss_h.item()}, loss_rec: {loss_rec.item()}, loss_z: {loss_z.item()}, loss_color: {0 if config.w_color==0 else loss_color.item()}")
				writer.add_scalar('loss', loss.item(), global_step)
				writer.add_scalar('loss_h', loss_h.item(), global_step)
				writer.add_scalar('loss_rec', loss_rec.item(), global_step)
				writer.add_scalar('loss_z', loss_z.item(), global_step)
				if config.w_color!=0:
					writer.add_scalar('loss_color', loss_color.item(), global_step)
			global_step += 1
		if epoch%100 == 0:
			x_enhance, h, z_gt, n_x = net(LI, HI)
			torchvision.utils.save_image(torchvision.utils.make_grid(x_enhance), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_output.png'))
			torchvision.utils.save_image(torchvision.utils.make_grid(h/2+0.5), os.path.join(config.snapshots_folder, 'sample', f'{epoch}h.png'))
			torchvision.utils.save_image(torchvision.utils.make_grid(LI), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_input.png'))
			torchvision.utils.save_image(torchvision.utils.make_grid(HI), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_GT.png'))
			torchvision.utils.save_image(torchvision.utils.make_grid(n_x), os.path.join(config.snapshots_folder, 'sample', f'{epoch}_n_x.png'))
		# scheduler.step()
		# save param
		if ((epoch+1) % config.snapshot_iter) == 0:
			torch.save(net.module.state_dict() if len(config.device)>1 else net.state_dict(), os.path.join(config.snapshots_folder, "Epoch" + str(epoch+config.start_epoch) + '.pth')) 		
	writer.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Input Parameters
	parser.add_argument('--LI_path', type=str, default="/data1/wjh/LOL_v2/Real_captured/train/input")
	parser.add_argument('--HI_path', type=str, default="/data1/wjh/LOL_v2/Real_captured/train/gt")
	parser.add_argument('--yaml_file', type=str, default="config/Enhancement_MIRNet_v2_Lol.yml")
	parser.add_argument('--dataset', type=str, default="lol", help='lol|EE')
	parser.add_argument('--model', type=str, default="ECNet", help='ZeroDCE or ECNet')
	parser.add_argument('--srnet', type=str, default="srnet", help='srnet|srnet_plain|srnet_rrdb|srnet_strideconv|srnet_deconv|mirnetv2|srnetdeeper|srnetwider|srnetdeeperres|srnetselfnorm')
	parser.add_argument('--ignet', type=str, default="ignet", help='ignet|ignet_group|ignet_group2|ignet_res')
	parser.add_argument('--csnet', type=str, default="csnet", help='csnet|csnet_res|csnet_dense|csnet_tanh')
	parser.add_argument('--adam', action='store_true', help='if not specified, using SGD instead')
	parser.add_argument('--no_detach', action='store_true', help='if specified, do not detach when computing realigning loss')
	parser.add_argument('--w_h', type=float, default=0.01)
	parser.add_argument('--w_z', type=float, default=0.1)
	parser.add_argument('--w_color', type=float, default=0.0)
	parser.add_argument('--lambda_', type=float, default=0.8)
	parser.add_argument('--warm_up', type=float, default=100.0)
	parser.add_argument('--w_eq', type=float, default=0.0)
	parser.add_argument('--lr', type=float, default=0.0003)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=3000)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--log_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=100)
	parser.add_argument('--snapshots_folder', type=str, default="/data1/wjh/ECNet/baseline/snapshot")
	parser.add_argument('--load_pretrain', action='store_true', help='load pretrained weights or not')
	parser.add_argument('--start_epoch', type=int, default=0)
	parser.add_argument('--l_rec_epoch', type=int, default=4000, help='using ssim loss')
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	parser.add_argument('--device', type=str, default= "0")
	config = parser.parse_args()
	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	print(config)
	train(config)
