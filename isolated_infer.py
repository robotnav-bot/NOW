# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#from distributed import init_distributed
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np
import glob

from diffusion import create_diffusion
from diffusion.diffusion_infer import model_forward_wrapper
from diffusers.models import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available

import distributed as dist
from models.unet_3d_condition import NOW
from PIL import Image
from distributed import init_distributed
from data.dataset import NavDataset
import json
from einops import rearrange
from data.data_utils import unnormalize
from data.data_utils import draw_traj_img
import matplotlib.pyplot as plt
from flowmatching import create_targets, short_cut_infer
from models.attention import TransformerBlock3D, TransformerBlock3D_Temporal
from models.resnet import ResnetBlock3D

import time

def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = unnormalize(img)
        
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)
    

def get_dataset_eval(config, dataset_name):
    data_config = config["eval_datasets"][dataset_name]    
    
    dataset = NavDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                normalize=config["normalize"],
                traj_stride=config["traj_stride"],
                traj_names="traj_names.txt",
                is_training = False#data_split_type=="train"
            )
    
    return dataset


def generate_rollout(args, output_dir, idxs, all_models, obs_image, gt_image, actions, device, bfloat_enable):
    rollout_stride = 1
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]
    curr_obs = obs_image.clone().to(device)

    latent_size = curr_obs.shape[3] // 8
    len_traj_pred = gt_image.shape[1]
    num_cond = obs_image.shape[1]

    model, diffusion, vae = all_models

    if args.gt:
        samples = gt_image.clone().to(device)
    else:
        if diffusion is not None:
            samples = model_forward_wrapper(all_models, obs_image, actions, \
                                num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                latent_size=latent_size, device=device, bfloat_enable=bfloat_enable)
        else:
            samples = short_cut_infer((model, diffusion, vae), obs_image, actions, \
                                num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                latent_size=latent_size, device=device, bfloat_enable=bfloat_enable)

        samples = rearrange(samples, 'b c f2 h w -> b f2 c h w')
            
    
    for i in range(gt_image.shape[1]):
        x_pred_pixels = samples[:, i]
        visualize_preds(output_dir, idxs, i, x_pred_pixels)

def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[batch_idx], True)

@torch.no_grad()
def main(args):
    print(args)
    device = torch.device("cuda:0")

    exp_eval = glob.glob(os.path.join(args.exp, "*.yaml"))[0]

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)



    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.exp)))
        args.save_output_dir = args.save_output_dir + "_%s"%(args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        
        with open(config["model_config"], "r") as f:
            model_config_dict = json.load(f)
        model_config_dict["diffusion_type"]=config["diffusion_type"]

        model = NOW.from_config(model_config_dict).to(device)


        print(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        ckp = torch.load(os.path.join(args.exp ,f'checkpoints/{args.ckp}.pth.tar'), map_location='cpu', weights_only=False)
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()
        model.to(device)

        if is_xformers_available():
            try:
                model.enable_xformers_memory_efficient_attention()
                print(
                    "enable_xformers_memory_efficient_attention"
                ) 
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        # model = torch.compile(model)
        if config["diffusion_type"]=="diffusion":
            diffusion = create_diffusion(str(250), learn_sigma=False) ### change steps
        elif config["diffusion_type"]=="shortcut":
            diffusion=None

        vae = AutoencoderKL.from_pretrained(config["pretrained_vae_path"],subfolder="vae").to(device)
        model_lst = (model, diffusion, vae)

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(config, dataset_name)


        curr_data_loader = torch.utils.data.DataLoader(
                            dataset_val, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False
                        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = dist.MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        
        for data_iter_step, (idxs, obs_image, target_image, actions) in enumerate(metric_logger.log_every(curr_data_loader, print_freq, header)):
            assert config['context_size'] == obs_image.shape[1], "Context_size must be equal to the frame num of obsimage."
            assert config['len_traj_pred'] == target_image.shape[1], "Len_traj_pred must be equal to the frame num of target_image."
            assert actions.shape[1] == obs_image.shape[1]+target_image.shape[1], "len(actions) should == len(obs_image)+len(target_image)."

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                obs_image = obs_image.to(device)
                gt_image = target_image.to(device)
                actions = actions.to(device)

                # for i in range(10):
                #     fig=draw_traj_img(obs_image[i]*0.5+0.5,target_image[i]*0.5+0.5,target_image[i]*0.5+0.5,actions[i])
                #     fig.savefig('show/'+str(i)+'.png')
                #     plt.close()

                num_cond = config["context_size"]
                curr_rollout_output_dir = os.path.join(dataset_save_output_dir, 'rollout')
                os.makedirs(curr_rollout_output_dir, exist_ok=True)
                generate_rollout(args, curr_rollout_output_dir, idxs, model_lst, obs_image, gt_image, actions, device, bfloat_enable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--exp", type=str, default="", help="experiment name")
    parser.add_argument("--ckp", type=str, default='latest')
    parser.add_argument("--datasets", type=str, default="habitat-ac26ZMwG7aT_for_test", help="dataset name")
    parser.add_argument("--num_workers", type=int, default=12, help="num workers")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    parser.add_argument("--bfloat16", type=int, default=1)
    args = parser.parse_args()
    
    
    main(args)