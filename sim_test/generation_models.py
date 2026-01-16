from models.unet_3d_condition import NOW
import torch
from diffusers.models import AutoencoderKL
from diffusion.diffusion_infer import model_forward_wrapper
from einops import rearrange
import yaml, json
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms as T
from PIL import Image
import numpy as np
import os

def get_generation_model(model_type, exp_eval, model_path):

    if model_type == "now":
        return NOW_MODEL(exp_eval, model_path)




class NOW_MODEL:
    def __init__(self, exp_eval, model_path):
        self.exp_eval=exp_eval
        self.model_path=model_path

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        config = default_config
        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)

        with open(config["model_config"], "r") as f:
            model_config_dict = json.load(f)
        model_config_dict["diffusion_type"]=config["diffusion_type"]
        self.device=torch.device("cuda:0")
        model = NOW.from_config(model_config_dict).to(self.device)

        ckp = torch.load(model_path, map_location='cpu', weights_only=False)
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()

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

        if config["diffusion_type"]=="diffusion":
            diffusion = create_diffusion(str(250), learn_sigma=False) ### change steps
        elif config["diffusion_type"]=="shortcut":
            diffusion=None

        vae = AutoencoderKL.from_pretrained(config["pretrained_vae_path"],subfolder="vae").to(self.device)
        model.eval()
        vae.eval()
        self.model_lst = (model, diffusion, vae)
            
        self.transform = T.Compose([
                    T.Resize((config["image_size"],config["image_size"])), #T.CenterCrop(args.resolution),
                    T.ToTensor(),
                ])
        self.config = config
    
    def _wrap_angle(self, theta):
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    def get_trajs(self, actions):

        trajs =  torch.zeros((actions.shape[0], actions.shape[1]+1, 3)).to(actions.device)

        for t in range(1, trajs.shape[1]):
            current_pose = trajs[:, t-1, :]
            current_action = actions[:, t-1, :]
            dx = current_action[:, 0]
            dtheta = current_action[:, 1]
            
            global_dx = dx * torch.cos(current_pose[:, 2])
            global_dy = dx * torch.sin(current_pose[:, 2])
            new_theta = self._wrap_angle(current_pose[:, 2] + dtheta)
            
            trajs[:, t, 0] = current_pose[:, 0] + global_dx
            trajs[:, t, 1] = current_pose[:, 1] + global_dy
            trajs[:, t, 2] = new_theta

        actions=torch.zeros((trajs.shape[0],trajs.shape[1],4))
        actions[:,:,:2]=trajs[:,:,:2]
        actions[:,:,2]=torch.cos(trajs[:,:,2])
        actions[:,:,3]=torch.sin(trajs[:,:,2])

        return actions

    def get_samples(self, obs_image, actions, num_cond=1, len_traj_pred=11, check_num = -1, bfloat_enable=True,visualize=False):

        actions=self.get_trajs(actions)
        
        from utils.flowmatching import create_targets, short_cut_infer
        model, diffusion, vae = self.model_lst
        latent_size = obs_image.shape[3] // 8
        B=actions.shape[0]

        obs_image=obs_image.repeat((B,1,1,1,1))

        if diffusion is not None:
            samples = model_forward_wrapper(self.model_lst, obs_image, actions, \
                                num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                latent_size=latent_size, device=obs_image.device, bfloat_enable=bfloat_enable)
        else:
            samples = short_cut_infer(self.model_lst, obs_image, actions, \
                                num_cond=num_cond, len_traj_pred=len_traj_pred, \
                                latent_size=latent_size, device=obs_image.device, bfloat_enable=bfloat_enable, \
                                visualize=visualize,check_num=check_num)

        samples = rearrange(samples, 'b c f2 h w -> b f2 c h w')

        return samples
