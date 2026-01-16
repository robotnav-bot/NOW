
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt 
import yaml
import json
from einops import rearrange

import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F

# from distributed import init_distributed
from models.unet_3d_condition import NOW
from models.attention import TransformerBlock3D, TransformerBlock3D_Temporal
from models.resnet import ResnetBlock3D

from diffusion import create_diffusion
from diffusion.diffusion_infer import model_forward_wrapper

from data.dataset import NavDataset
from data.data_utils import draw_traj_img
from torch.cuda.amp import GradScaler
from flowmatching import create_targets, short_cut_infer
import copy
import datetime
import shutil
import re
import torch.nn as nn
import math

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # if dist.get_rank() == 0:  # real logger
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    # else:  # dummy logger (does nothing)
    #     logger = logging.getLogger(__name__)
    #     logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = torch.device("cuda:0")
    seed = args.global_seed 
    torch.manual_seed(seed)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['results_dir'], f"{config['run_name']}_{timestamp}")

    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    config_copy_path = os.path.join(experiment_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_copy_path)
    logger.info(f"Config file copied to {config_copy_path}")

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(config["pretrained_vae_path"],subfolder="vae").to(device)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']

    with open(config["model_config"], "r") as f:
        model_config_dict = json.load(f)
    model_config_dict["diffusion_type"]=config["diffusion_type"]

    model = NOW.from_config(model_config_dict)#.to(device)


    ema = deepcopy(model)#.to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # loading from the pretrained checkpoint
    if config["pretrain"]:
        assert config.get('pretrained_checkpoint', 0) != 0, "a pretrained_checkpoint is needed when finetuning"

        ckp_path = config['pretrained_checkpoint']
        logger.info(f"Loading model from {ckp_path}")

        ckp = torch.load(ckp_path, map_location='cpu', weights_only=False)
        model_ckp = {k.replace('_orig_mod.', ''):v for k,v in ckp['model'].items()}
        res = model.load_state_dict(model_ckp, strict=False)
        logger.info(f"Loading model weights {res}")

        model_ckp = {k.replace('_orig_mod.', ''):v for k,v in ckp['ema'].items()}
        res = ema.load_state_dict(model_ckp, strict=False)
        logger.info(f"Loading EMA model weights {res}")

        requires_grad(model, True)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, weight_decay=0)


    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = GradScaler()


    start_epoch = 0
    train_steps = 0


        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)

    if is_xformers_available():
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info(
                "enable_xformers_memory_efficient_attention"
            ) 
            print("enable_xformers_memory_efficient_attention", flush=True)
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    # model = DDP(model, device_ids=[device])
    model = model.to(device)
    ema = ema.to(device)

    if config["diffusion_type"]=="diffusion":
        diffusion = create_diffusion(timestep_respacing="",learn_sigma=False)  # default: 1000 steps, linear noise schedule
    elif config["diffusion_type"]=="shortcut":
        diffusion=None
    logger.info(f"model based on stable diffusion with temp attn Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = []
    test_dataset = []

    if "traj_stride" not in config:
        config["traj_stride"]=1 #use all trajs when training
        
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                if "len_traj_pred" in data_config:
                    len_traj_pred=data_config["len_traj_pred"]
                else:
                    len_traj_pred=config["len_traj_pred"]

                dataset = NavDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    len_traj_pred=len_traj_pred,
                    context_size=config["context_size"],
                    normalize=config["normalize"],
                    traj_stride=config["traj_stride"],
                    traj_names = "traj_names.txt",
                    random_traj=config["random_traj"],
                    is_training = True#data_split_type=="train"
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    test_dataset.append(dataset)
                print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")
    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    f1=config['context_size']
    f2=config['len_traj_pred']
    f=f1+f2

    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for obs_image, target_image, actions in loader:
            """
            obs_image:  [B,context_size,3,image_size,image_size]
            target_image: [B,len_traj_pred,3,image_size,image_size]
            actions: [B,len_traj_pred+1,4]
            """
            obs_image = obs_image.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            assert config['context_size'] == obs_image.shape[1], "Context_size must be equal to the frame num of obsimage."
            assert config['len_traj_pred'] == target_image.shape[1], "Len_traj_pred must be equal to the frame num of target_image."
            assert actions.shape[1] == obs_image.shape[1]+target_image.shape[1], "len(actions) should == len(obs_image)+len(target_image)."
            
            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B = obs_image.shape[0]
                    images = torch.cat([obs_image,target_image],dim=1)
                    images = images.flatten(0,1)
                    latents = tokenizer.encode(images).latent_dist.sample().mul_(0.18215)
                    latents = latents.unflatten(0, (B, f))
                # The shape of the sample sent to the model is (b c f h w) 
                latents=latents.permute(0,2,1,3,4)
                latents_obs=latents[:,:,:f1,:,:]
                latents_target=latents[:,:,f1:,:,:]
                
                if config["diffusion_type"]=="diffusion":
                    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
                    model_kwargs = dict(cond_sample=latents_obs,actions=actions,cond_frame=f1)
                    loss_dict = diffusion.training_losses(model, latents_target, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                elif config["diffusion_type"]=="shortcut":
                    x_t, v_t, t, dt_base, actions_, latents_x0_, _= \
                        create_targets(latents_target, latents_obs, actions, copy.deepcopy(model))
                    v_prime, _ = model.forward(sample=x_t, timestep=t, cond_sample=latents_x0_, actions=actions_,\
                                            d=dt_base, cond_frame=f1)

                    loss = F.mse_loss(v_prime, v_t, reduction="none").mean([1, 2, 3, 4]).mean()


            if not bfloat_enable:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()
            
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = B*steps_per_sec #dist.get_world_size()*B*steps_per_sec
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()# / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),#model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps
                }
                if bfloat_enable:
                    checkpoint.update({"scaler": scaler.state_dict()})
                checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                torch.save(checkpoint, checkpoint_path)
                if train_steps % (10*args.ckpt_every) == 0 and train_steps > 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond, len_traj_pred)
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")


@torch.no_grad()
def evaluate(model, vae, diffusion, test_dataloaders, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond, len_traj_pred):

    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    import sys, os
    sys.path.insert(0, "/root/workspace/code/Navigation_worldmodel/weights/dreamsim")

    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True, cache_dir="/root/workspace/code/Navigation_worldmodel/weights/dreamsim")
    

    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for obs_image, target_image, actions in loader:
        obs_image = obs_image.to(device)
        target_image = target_image.to(device)
        actions = actions.to(device)
        with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
            B = obs_image.shape[0]

            if diffusion is not None:
                samples = model_forward_wrapper((model, diffusion, vae), obs_image, actions, \
                                    num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                    latent_size=latent_size, device=device, bfloat_enable=bfloat_enable)
            else:
                samples = short_cut_infer((model, diffusion, vae), obs_image, actions, \
                                    num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                    latent_size=latent_size, device=device, bfloat_enable=bfloat_enable)
                
            samples = samples * 0.5 + 0.5 # (b,c,f2,h,w)
            obs_image = obs_image * 0.5 + 0.5 # (b,f1,c,h,w)
            target_image = target_image * 0.5 + 0.5 # (b,f2,c,h,w)

            samples_ = rearrange(samples, 'b c f2 h w -> (b f2) c h w')
            target_image_ = rearrange(target_image, 'b f2 c h w -> (b f2) c h w')

            samples_ = samples_.to(torch.float32)
            target_image_ = target_image_.to(torch.float32)

            samples_=F.interpolate(samples_, size=(224, 224), mode='bilinear', align_corners=False)
            target_image_=F.interpolate(target_image_, size=(224, 224), mode='bilinear', align_corners=False)
            
            samples_ = samples_.to(torch.float16)
            target_image_ = target_image_.to(torch.float16)

            res = eval_model(target_image_, samples_)
            score += res.sum()
            n_samples += len(res)
        break
    
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(samples.shape[0], 10)):

        fig=draw_traj_img(obs_image[i],samples[i].permute(1,0,2,3),target_image[i],actions[i])
        fig.savefig(f'{save_dir}/{i}.png')
        plt.close()

    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_shortcut_w_pretrain.yaml")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=0)
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
