import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import copy
sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils_eval import *
import yaml
import cv2
from skimage import io
import glob

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="PECNET_social_model1.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")
parser.add_argument('--person_id', '-pi', type=int, default=1)

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def test(test_dataset, t, model, origin, num, fnum, best_of_n = 1):

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0
	with torch.no_grad():
		
		for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			#x = traj[:, num:num+hyper_params["past_length"], :]
			# reshape the data
			x = torch.DoubleTensor(t).to(device)
			x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
			x = x.to(device)

			for index in range(best_of_n):
				dest_recon = model.forward(x, initial_pos, device=device)
				dest_recon = dest_recon.cpu().numpy()

			best_guess_dest = dest_recon

			# back to torch land
			best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

			# using the best guess for interpolation
			interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
			interpolated_future = interpolated_future.cpu().numpy()
			best_guess_dest = best_guess_dest.cpu().numpy()


			# final overall prediction
			predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
			predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
			print("../../../sort/output/3D/image{0:06d}.jpg".format(fnum+7))
			img1 = np.asarray(io.imread("../../../sort/output/3D/image{0:06d}.png".format(fnum+7)))
			x /= hyper_params["data_scale"]
			pf = predicted_future / hyper_params["data_scale"]
			for j in range(len(x)):
				for k in range(0, 16,2):
					cv2.circle(img1, (int(x[j][k]+origin[0]), int(x[j][k+1]+origin[1])), 1,(0,255,0), 5)
				for k in range(12):
					cv2.circle(img1, (int(pf[j][k][0]+origin[0]), int(pf[j][k][1]+origin[1])), 1,(0,0,255), 5)
			cv2.imwrite(f'./result/image{0:06d}.png'.format(fnum+7), img1)

def main():
	N = args.num_trajectories #number of generated trajectories
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	#test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
	test_dataset = SocialDataset(set_name="test", b_size=args.person_id, t_tresh=0, d_tresh=25, verbose=args.verbose)

	'''for traj in test_dataset.trajectory_batches:
		origin = copy.deepcopy(traj[:, :1, :]).squeeze()
		traj -= traj[:, :1, :]
		traj *= hyper_params["data_scale"]

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 5
	fnum = int(test_dataset.frame_num[0][0])
	for i in range(int(test_dataset.frame_num[0][-1]-test_dataset.frame_num[0][0])-7):
		test(test_dataset, model, origin, num=i, fnum = fnum, best_of_n = N)
		fnum += 1'''

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 5
	fnum = int(test_dataset.frame_num[0][0])
	for i in range(int(test_dataset.frame_num[0][-1]-test_dataset.frame_num[0][0])-16):
		for traj in test_dataset.trajectory_batches:
			t = copy.deepcopy(traj[:, i:i+8, :])
			print(t)
			origin = copy.deepcopy(t[:, :1, :]).squeeze()
			t -= t[:, :1, :]
			t *= hyper_params["data_scale"]
		test(test_dataset, t, model, origin, num=i, fnum = fnum, best_of_n = N)
		fnum += 1
	
	# 영상으로 제작	
	img_array = []
	size = (512, 512)
	file_list = glob.glob('./result/*.jpg')
	file_list = sorted(file_list, key=lambda x: int(x[9:-4]))
	for filename in file_list:
		print(filename)
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)
	
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	out = cv2.VideoWriter(f'output_{args.person_id}.mp4', fourcc, 5, size, True)
	
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	

main()
