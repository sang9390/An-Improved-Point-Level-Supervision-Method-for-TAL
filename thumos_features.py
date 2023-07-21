import torch.utils.data as data
import os
import csv
import json
import numpy as np
import pandas as pd
import torch
import pdb
import time
import random
import utils
import config

### start###
def json_reader(dir):
    # input : 읽을 json file의 dir(string)를 입력으로 받음
    # output: json file을 불러옴
    with open(dir, 'r') as f:
        ann = json.load(f)
    return ann
### end###


class ThumosFeature(data.Dataset):
    # def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, seed=-1, supervision='point'):    original_thumos_features.py_001
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, seed=-1, supervision='point', bkg_ann_dir=None):  # modification_thumos_features.py_001
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.sampling = sampling
        self.supervision = supervision

        ### strat###
        self.bkg_ann_dir = bkg_ann_dir
        if not bkg_ann_dir == None:
            self.bkg_labels = json_reader(bkg_ann_dir)
        ### end###

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        self.fps_dict = json.load(open(os.path.join(data_path, 'fps_dict.json')))

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())
        self.num_classes = len(self.class_name_to_idx.keys())

        if self.supervision == 'point':
            self.point_anno = pd.read_csv(os.path.join(data_path, 'point_gaussian', 'point_labels.csv'))

        self.stored_info_all = {'new_dense_anno': [-1] * len(self.vid_list), 'sequence_score': [-1] * len(self.vid_list)}

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        # label, point_anno = self.get_label(index, vid_num_seg, sample_idx)    #orginal_thumos_features.py_002
        ### start###  modification_thumos_features.py_002
        if self.bkg_ann_dir == None:
            label, point_anno  = self.get_label(index, vid_num_seg, sample_idx)
        else:
            label, point_anno, bkg_ann = self.get_label(index, vid_num_seg, sample_idx)
        ### end###

        stored_info = {'new_dense_anno': self.stored_info_all['new_dense_anno'][index], 'sequence_score': self.stored_info_all['sequence_score'][index]}

        # return index, data, label, point_anno, stored_info, self.vid_list[index], vid_num_seg #orginal_thumos_features.py_003
        ### start###    modification_thumos_features.py_003
        if self.bkg_ann_dir == None:
            return index, data, label, point_anno, stored_info, self.vid_list[index], vid_num_seg, 1
        else:
            return index, data, label, point_anno, stored_info, self.vid_list[index], vid_num_seg, bkg_ann
        ### end###

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'video':
            return label, torch.Tensor(0)

        elif self.supervision == 'point':

            ### start###
            act_frm_index = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
            bkg_frm_index = np.zeros([vid_num_seg, 1], dtype=np.float32)
            ### end###

            temp_anno = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
            t_factor = self.feature_fps / (self.fps_dict[vid_name] * 16)

            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class_index']]

            ### strat###
            act_weight = []
            ### end###

            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = temp_df['class_index'][key]

                temp_anno[int(point * t_factor)][class_idx] = 1

                ### start###
                act_frm_index[int(point * t_factor)][class_idx] = point
                act_weight.append(point * t_factor)
                ### end###

            point_label = temp_anno[sample_idx, :]

            # return label, torch.from_numpy(point_label)   original_thumos_features.py_004
            ### start###    modification_thumos_features.py_004
            if self.bkg_ann_dir == None:
                return label, torch.from_numpy(point_label)
            else:
                bkg_label = self.bkg_labels[vid_name]
                bkg_temp_anno = np.zeros([vid_num_seg, 1], dtype=np.float32)
                for i in bkg_label:
                    if int(i * t_factor) >= bkg_temp_anno.shape[0]:
                        continue
                    else:
                        bkg_temp_anno[int(i * t_factor)] = 1
                        bkg_frm_index[int(i * t_factor)] = i
                bkg_label = bkg_temp_anno[sample_idx, :]

                overlap_index = self.overlap_point_search(point_label, bkg_label)
                try:
                    if overlap_index[0].item() > -1:
                        bkg_label = self.overlap_bkg_move(act_frm_index, bkg_frm_index, overlap_index, bkg_label)
                except:
                    _ = 0

                return label, torch.from_numpy(point_label), torch.from_numpy(bkg_label)
            ### end###


    def random_perturb(self, length):
        if self.num_segments == length or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if length <= self.num_segments or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    ### start###
    def overlap_point_search(self, act_label, bkg_label):
        point_label_for = act_label.sum(axis=1)
        point_label_for_ = np.where(point_label_for >= 1, 1, point_label_for)
        point_ann_overlap = point_label_for_ + bkg_label[:, 0]
        overlap_index = np.where(point_ann_overlap >= 2)

        return overlap_index

    def overlap_bkg_move(self, act_frm_index, bkg_frm_index, overlap_index, bkg_point_label):
        act_frm_index = act_frm_index.sum(axis=1) # action segment가 겹친다면 값이 비정상적일 수 있는 우려가 있기는 함
        for i in overlap_index:
            if act_frm_index[i] > bkg_frm_index[i, 0]:
                if i - 1 == 0:
                    bkg_point_label[i, 0] = 0
                    continue
                if act_frm_index[i - 1] > 0:
                    bkg_point_label[i, 0] = 0
                    continue
                bkg_point_label[i, 0] = 0
                bkg_point_label[i - 1, 0] = 0
            else:
                if act_frm_index[i + 1] > 0:
                    bkg_point_label[i, 0] = 0
                    continue
                bkg_point_label[i, 0] = 0
                bkg_point_label[i + 1, 0] = 1

        return bkg_point_label
    ### end###