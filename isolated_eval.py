# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import json

from PIL import Image

# Eval
import lpips
from dreamsim import dreamsim
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
import distributed as dist


def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == 'lpips':
        general_lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(device) # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
                
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = general_lpips_loss_fn.forward(all_img0, all_img1)
            dist_avg = dist.mean()
            
            return dist_avg
    elif loss_fn_type == 'dreamsim':
        # dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, cache_dir="/root/workspace/d15f4efhri0c73dr6hsg/weights/dreamsim",\
        #                                         device=device)

        import sys, os
        sys.path.insert(0, "/root/workspace/code/Navigation_worldmodel/weights/dreamsim")

        from dreamsim import dreamsim
        dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, cache_dir="/root/workspace/code/Navigation_worldmodel/weights/dreamsim")
        

        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = preprocess(Image.open(img0_path)).to(device)
                img1 = preprocess(Image.open(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
            
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = dreamsim_loss_fn(all_img0, all_img1)
            dist_mean = dist.mean()
            
            return dist_mean
    elif loss_fn_type == 'fid':
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)
        
        return fid_metrics
    else:
        raise NotImplementedError
    
    return loss_fn


def evaluate(args, dataset_name, eval_name, metric_logger, loss_fns, gt_dir, exp_dir, secs):
    lpips_loss_fn, dreamsim_loss_fn, fid_loss_fn = loss_fns
    
    image_idxs = secs.copy()
        
    eps = os.listdir(gt_dir)
    
    for batch_start in tqdm(range(0, len(eps), args.batch_size), total=(len(eps) + args.batch_size - 1) // args.batch_size):
        batch_eps = eps[batch_start:batch_start + args.batch_size]
        
        gt_batch, exp_batch = {}, {}
        gt_paths_batch, exp_paths_batch = {}, {}
        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []
            gt_paths_batch[sec] = []
            exp_paths_batch[sec] = []
        
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)
        
            if not os.path.isdir(gt_ep_dir) and not os.path.isdir(exp_ep_dir):
                continue
        
            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f'{image_idx}.png')
                gt_sec_img = transforms.ToTensor()(Image.open(gt_sec_img_path).convert("RGB")).unsqueeze(0)
                exp_sec_img_path = os.path.join(exp_ep_dir, f'{image_idx}.png')
                exp_sec_img = transforms.ToTensor()(Image.open(exp_sec_img_path).convert("RGB")).unsqueeze(0)
                
                gt_batch[sec].append(gt_sec_img)
                gt_paths_batch[sec].append(gt_sec_img_path)
                exp_batch[sec].append(exp_sec_img)
                exp_paths_batch[sec].append(exp_sec_img_path)
            
        for sec in secs:
            lpips_dists = lpips_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            dreamsim_dists = dreamsim_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            
            metric_logger.meters[f'{dataset_name}_{eval_name}_lpips_{sec}s'].update(lpips_dists, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_dreamsim_{sec}s'].update(dreamsim_dists, n=1)
            
            sec_gt_batch = torch.cat(gt_batch[sec], dim=0)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0)
            
            fid_loss_fn[sec].update(images=sec_gt_batch, is_real=True)
            fid_loss_fn[sec].update(images=sec_exp_batch, is_real=False)
            
    for sec in secs:
        metric_logger.meters[f'{dataset_name}_{eval_name}_fid_{sec}s'].update(fid_loss_fn[sec].compute().item(), n=1)
        
def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    log_stats = {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}
    with open(log_p, 'w') as json_file:
        json.dump(log_stats, json_file, indent=4)  # indent=4 adds indentation for readability            


def main(args):
    device = 'cuda'
          
    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    secs = np.array([i for i in range(0, args.num_sec_eval)])
    
    # These loss functions do not accumulate
    lpips_loss_fn = get_loss_fn('lpips', secs, device)
    dreamsim_loss_fn = get_loss_fn('dreamsim', secs, device)

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        metric_logger = dist.MetricLogger(delimiter="  ")
        print("Evaluating rollout", dataset_name)
        # Rollout (LPIPS, DreamSim, FID)
        eval_name = 'rollout'
        gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
        exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
        rollout_fid_loss_fn = get_loss_fn('fid', secs, device)
        rollout_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, rollout_fid_loss_fn)
        with torch.no_grad():
            evaluate(args, dataset_name, eval_name, metric_logger, rollout_loss_fns, gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs)
        output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
        save_metric_to_disk(metric_logger, output_fn)


     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--gt_dir", type=str, default="output/gt", help="gt directory")
    parser.add_argument("--exp_dir", type=str, default="output/shortcut_w_pretrain_20260114_161940_latest", help="experiment directory")
    parser.add_argument("--num_sec_eval", type=int, default=11, help="experiment name")
    parser.add_argument("--datasets", type=str, default="", help="dataset name")
    
    parser.add_argument("--input_fps", type=int, default=4, help="experiment name")
    
    args = parser.parse_args()
    
    
    
    main(args)