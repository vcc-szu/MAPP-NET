'''
	ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''
import os
import os.path
import json
import numpy as np
import sys
import re

import torch
import torch.utils.data as pthdata


from mappnet.config import config

def np2pth(nparray, dtype):
	if isinstance(nparray, int) or isinstance(nparray, float):
		nparray = np.array(nparray)
	if dtype=='float':
		return torch.from_numpy(nparray.copy()).type(torch.FloatTensor)
	if dtype=='long':
		return torch.from_numpy(nparray.copy()).type(torch.LongTensor)
class MAPPNetDataset(pthdata.Dataset):
	def __init__(self, 
	split='train', 
	config = config):
		super().__init__()

		#self.data_path = config.data_config_path
		self.batch_size = config.batch_size
		self.num_point = config.num_point
		self.num_frame = config.num_frame
		
		self.shape_cat = {}
		self.shape_cat['train'] = [line.rstrip().split(' ') for line in open(config.train_datalist_path)]
		self.shape_cat['test'] = [line.rstrip().split(' ') for line in open(config.test_datalist_path)]
		assert(split=='train' or split=='test')
		self.classes = dict(self.shape_cat[split])
		self.classes_dict = {'T':0,'R':1,'TR':2}
		self.seg_classes = {'Ref':[0], 'Mov':[1]}        
		shape_names = [x[0] for x in self.shape_cat[split]]

		# list of (shape_name, shape_txt_file_path) tuple
		self.datapath = [(self.shape_cat[split][i][0], os.path.join(config.pc_path, shape_names[i])+'.pts') for i in range(len(self.shape_cat[split]))]
		self.segpath = [(self.shape_cat[split][i][0], os.path.join(config.seg_path, shape_names[i][:-6])+'.seg') for i in range(len(self.shape_cat[split]))]
		self.mopath = [(self.shape_cat[split][i][0], os.path.join(config.mo_path  , shape_names[i][:-6])+'.txt') for i in range(len(self.shape_cat[split]))]
		
		self.cache_size = config.cache_size # how many data points to cache in memory
		self.cache = {} # from index to (pc, cls) tuple

	def __getitem__(self, index): 
		if index in self.cache:
			pc, cls, seg, mo, pc_target = self.cache[index]
		else:
			pc = np.loadtxt(self.datapath[index][1],delimiter=' ').astype(np.float64)
			pc_target = np.zeros((self.num_frame, self.num_point, 3), dtype=np.float64)
			cur_frame_idx = int(self.datapath[index][1][-5])

			for i in range(self.num_frame):
				pc_target[i,:,:] = np.loadtxt(self.datapath[index][1][:-5]+str(cur_frame_idx+i+1)+'.pts',delimiter=' ').astype(np.float64)

			cls = self.classes[self.datapath[index][0]]
			cls = self.classes_dict[cls]

			seg = np.loadtxt(self.segpath[index][1]).astype(np.int64)

			# (2,3)[[posx,posy,posz]
			# 		[dirx,diry,dirz]]
			mo = np.loadtxt(self.mopath[index][1]).astype(np.float64)
			mo = mo.flatten()

			if len(self.cache) < self.cache_size:
				self.cache[index] = (pc, cls, seg, mo, pc_target)
		#return np2pth(pc,'float'), np2pth(cls,'long'), np2pth(seg,'long'), np2pth(mo,'float'), np2pth(pc_target,'float')
		dataterm = {'pc_in':np2pth(pc,'float'),
					'motion_type':np2pth(cls,'long'),
					'seg':np2pth(seg,'long'), 
					'motion_param':np2pth(mo,'float'),
					'pc_target':np2pth(pc_target,'float'),
					'index':np2pth(index, 'long')}
		return dataterm
	def __len__(self):
		return len(self.datapath)

	def get_batch(self, idxs, start_idx, end_idx):
		bsize = end_idx-start_idx
		batch_pc = np.zeros((bsize, self.num_point, 3))
		batch_pc_target = np.zeros((bsize, self.num_frame, self.num_point, 3))
		batch_cls = np.zeros((bsize), dtype=np.int64)
		batch_seg = np.zeros((bsize, self.num_point), dtype=np.int64)
		batch_mo = np.zeros((bsize, 6), dtype=np.float64)
		for i in range(bsize):
			pc, cls, seg, mo, pc_target = self.__getitem__(idxs[i+start_idx])
			batch_pc[i,:,:] = pc
			batch_pc_target[i,:,:,:] = pc_target
			batch_cls[i] = cls
			batch_seg[i,:] = seg
			batch_mo[i] = mo
		return batch_pc, batch_pc_target, batch_mo, batch_cls, batch_seg

	def get_name(self, idxs):
		# use to get corresponding names in testing
		return self.shape_cat['test'][idxs][0]



if __name__ == '__main__':
	d = MotionDataset(split='train')
	print(len(d))



