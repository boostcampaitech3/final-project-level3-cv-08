from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np

'''for sanity check'''
def naive_social(p1_key, p2_key, all_data_dict):
	if abs(p1_key-p2_key)<4:
		return True
	else:
		return False

def find_min_time(t1, t2):
	'''given two time frame arrays, find then min dist (time)'''
	min_d = 9e4
	t1, t2 = t1[:8], t2[:8]

	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
	'''given two time frame arrays, find then min dist'''
	min_d = 9e4
	p1x, p1y = p1x[:8], p1y[:8]
	p2x, p2y = p2x[:8], p2y[:8]

	for i in range(len(p1x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
	p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True

def mark_similar(mask, sim_list):
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1


def collect_data(set_name, dictionary, dataset_type = 'image', batch_size=20, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):

	assert set_name in ['train','val','test']

	'''Please specify the parent directory of the dataset. In our case data was stored in:
		root_path/trajnet_image/train/scene_name.txt
		root_path/trajnet_image/test/scene_name.txt
	'''

	rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)

	full_dataset = []
	full_masks = []

	current_batch = []
	mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

	current_size = 0
	social_id = 0

	data_by_id = dictionary.copy()
	all_data_dict = data_by_id.copy()
	if verbose:
		print("Total People: ", len(list(data_by_id.keys())))
	while len(list(data_by_id.keys()))>0:
		related_list = []
		curr_keys = list(data_by_id.keys())
		print(curr_keys)

		if current_size<batch_size:
			pass
		else:
			full_dataset.append(current_batch.copy())
			mask_batch = np.array(mask_batch)
			full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])

			current_size = 0
			social_id = 0
			current_batch = []
			mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

		current_batch.append((all_data_dict[curr_keys[0]]))
		related_list.append(current_size)
		current_size+=1
		del data_by_id[curr_keys[0]]

		for i in range(1, len(curr_keys)):
			#if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):
			current_batch.append((all_data_dict[curr_keys[i]]))
			related_list.append(current_size)
			current_size+=1
			del data_by_id[curr_keys[i]]

		mark_similar(mask_batch, related_list)
		social_id +=1


	full_dataset.append(current_batch)
	mask_batch = np.array(mask_batch)
	full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
	return full_dataset, full_masks

def generate_pooled_data(b_size, t_tresh, d_tresh, dictionary, train=True, scene=None, verbose=True):
	if train:
		full_train, full_masks_train = collect_data("train", dictionary, batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		train = [full_train, full_masks_train]
		train_name = "../social_pool_data/train_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)
		with open(train_name, 'wb') as f:
			pickle.dump(train, f)

	if not train:
		full_test, full_masks_test = collect_data("test", dictionary, batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		test = [full_test, full_masks_test]
		return test


def initial_pos(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

def calculate_loss(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)

	ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

class SocialDataset(data.Dataset):

	def __init__(self, dictionary, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=True):
		'Initialization'
		data = generate_pooled_data(25, 0,25, dictionary, train=False, verbose=True)#, root_path="./")
		traj, masks = data
		traj_new = []
		
		if id==False:
			for t in traj:
				t = np.array(t)
				#frame_num = list(map(max, t[:,:,1]))
				frame_num = t[:,:,1].copy()
				t = t[:,:,2:]
				traj_new.append(t)
				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)
		else:
			for t in traj:
				t = np.array(t)
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)


		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)

		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)
		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.frame_num = frame_num.copy()
		self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches)) #for relative positioning
		if verbose:
			print("Initialized social dataloader...")

"""
We've provided pickle files, but to generate new files for different datasets or thresholds, please use a command like so:
Parameter1: batchsize, Parameter2: time_thresh, Param3: dist_thresh
"""
# pickle dataset 생성하는 함수 dictionary: 
