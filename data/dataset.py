# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import yaml
import pickle
import tqdm
from torch.utils.data import Dataset
from .data_utils import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords, calculate_sin_cos, CenterCropAR
from torchvision import transforms
import random
import re

def get_uni_sequence(max_goal_dist, len_traj_pred):
    assert len_traj_pred <= max_goal_dist
    
    step = max_goal_dist / (len_traj_pred + 1)
    
    selected = [int(i * step) for i in range(1, len_traj_pred + 1)]
    
    selected.sort()  
    return selected

def get_random_sequence(max_goal_dist, len_traj_pred):
    assert len_traj_pred <= max_goal_dist
    numbers = list(range(1, max_goal_dist + 1))
    sampled = random.sample(numbers, len_traj_pred)
    sampled.sort()  
    return sampled

class NavDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        traj_stride: int, 
        traj_names: str = "traj_names.txt",
        len_traj_pred: int = 11,
        context_size: int = 1,
        normalize: bool = False,
        is_training: bool = True,
        random_traj=True,
        train_pos_decoder: bool = False
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name



        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        with open("config/data_config.yaml", "r") as f:
            all_data_config = yaml.safe_load(f)

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transforms.Compose([
                        CenterCropAR(),
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
        
        self.max_dist_cat = 20

        self._load_index()
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

        self.is_training = is_training
        self.random_traj = random_traj
        self.train_pos_decoder = train_pos_decoder

    def _load_index(self) -> None:

        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_n{self.context_size}_len_traj_pred_{self.len_traj_pred}_traj_stride_{self.traj_stride}_max_goal_distance_{self.max_dist_cat}.pkl",
        )
        if os.path.exists(index_to_data_path):
            print("Index file already exist, load directly")
            with open(index_to_data_path, "rb") as f:
                self.index_to_data = pickle.load(f)
        else:
            print("Building an index")
            self.index_to_data = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump(self.index_to_data, f)

    def _build_index(self, use_tqdm: bool = False):
        samples_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index
  
    def _get_trajectory(self, trajectory_name):

        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)

        if "habitat" in self.dataset_name:
            if isinstance(traj_data, list):
                traj_data = np.array(traj_data)

            traj_data_new = np.zeros_like(traj_data)
            traj_data_new[:,0] = -traj_data[:,1]
            traj_data_new[:,1] = -traj_data[:,0]
            traj_data_new[:,2] =  traj_data[:,2]

            traj_data={}
            traj_data["position"]=traj_data_new[:,:2]
            traj_data["yaw"]=traj_data_new[:,2]
        else:
            for k,v in traj_data.items():
                traj_data[k] = v.astype('float')

        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, target_times):
        """
        target_times: List[int]
        """
        indexes=np.array([curr_time]+target_times)
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1

        yaw = traj_data["yaw"][indexes]
        positions = traj_data["position"][indexes]
        # yaw = traj_data["yaw"][start_index:end_index]
        # positions = traj_data["position"][start_index:end_index]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)        
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        return actions

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, max_goal_distance = self.index_to_data[i]

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            
            if self.is_training:
                # for train, make the target distance changeable
                if self.random_traj:
                    target_times = get_random_sequence(max_goal_distance,self.len_traj_pred)
                    target_times = [t+curr_time for t in target_times]
                else:
                    target_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            else:
                # just for test, make the distance the same
                # target_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
                target_times = get_uni_sequence(max_goal_distance,self.len_traj_pred)
                target_times = [t+curr_time for t in target_times]

            
            context = [(f_curr, t) for t in context_times]
            target = [(f_curr, t) for t in target_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t)).convert('RGB')) for f, t in context])
            target_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t)).convert('RGB')) for f, t in target])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions = self._compute_actions(curr_traj_data, curr_time, target_times) # last argument is dummy goal
            # actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            # delta = get_delta_np(actions)
            actions=actions.astype(np.float32)

            actions = torch.as_tensor(actions, dtype=torch.float32)
            actions = calculate_sin_cos(actions)
            # actions=actions[1:,:]

            if not self.is_training:
                return (
                    torch.tensor([i], dtype=torch.float32), # for logging purposes when eval
                    torch.as_tensor(obs_image, dtype=torch.float32),
                    torch.as_tensor(target_image, dtype=torch.float32),
                    actions
                )
            else:
                if self.train_pos_decoder: # output the ground truth pos
                    return (
                        torch.as_tensor(obs_image, dtype=torch.float32),
                        torch.as_tensor(target_image, dtype=torch.float32),
                        torch.as_tensor(curr_traj_data["position"][curr_time], dtype=torch.float32)
                    )
                else:
                    return (
                        torch.as_tensor(obs_image, dtype=torch.float32),
                        torch.as_tensor(target_image, dtype=torch.float32),
                        actions
                    )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
        
        