from einops import rearrange, repeat, reduce
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from utils.ddim_sampling_utils import ddim_sample, save_visualization_onegif, short_cut_infer_onestep
from utils.flowmatching import create_targets, short_cut_infer
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from data.data_utils import draw_traj_img
import os
from PIL import Image
import logging
import cv2

def save_obs_and_selected_preds(obs_image, samples, save_path, line_thickness=10):
  
    obs = obs_image.detach().cpu()
    if obs.dim() == 4: 
        obs = obs[0]  # (1, 3, H, W) -> (3, H, W)
    
    obs = torch.clamp(obs * 0.5 + 0.5, 0, 1)
    
    samples = samples.detach().cpu()
    samples = torch.clamp(samples * 0.5 + 0.5, 0, 1)

    _, H, W = obs.shape
    
    green_line = torch.zeros(3, H, line_thickness)
    green_line[1, :, :] = 1.0  # R=0, G=1, B=0
    
    black_line = torch.ones(3, H, line_thickness)

    stitch_list = [obs, green_line]
    
    target_indices = [0, 5, 10]
    
    valid_indices = [idx for idx in target_indices if idx < samples.shape[0]]
    
    for i, idx in enumerate(valid_indices):
        frame = samples[idx]
        stitch_list.append(frame)
        
        if i < len(valid_indices) - 1:
            stitch_list.append(black_line)

    # ------------------------------------------------
    final_tensor = torch.cat(stitch_list, dim=2)

    # ------------------------------------------------
    # (3, H, W) -> (H, W, 3)
    final_img_np = final_tensor.permute(1, 2, 0).numpy()
    
    final_img_uint8 = (final_img_np * 255).astype(np.uint8)
    
    final_img_bgr = cv2.cvtColor(final_img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, final_img_bgr)




def plot_trajectory(traj):
    plt.plot(traj[:,0], traj[:,1], 'b-o')
    plt.quiver(traj[:,0], traj[:,1], 
               np.cos(traj[:,2]), np.sin(traj[:,2]))
    plt.axis('equal')
    plt.savefig("traj.png")


def crop_depth(depth,crop_rate=0.5):

    h, w = depth.shape[1:]
    crop_size = 200
    h_start = h // 2 - crop_size // 2
    h_end = h_start + crop_size
    w_start = w // 2 - crop_size // 2
    w_end = w_start + crop_size

    center_region = depth[:, h_start:h_end, w_start:w_end]
    mean_depth = center_region.min(axis=(1, 2))

    return mean_depth

class CEMTrajectoryPlanner:
    def __init__(self, 
                 goal,
                 lpips_fn=None,
                 dist_model=None,
                 DINO=None,
                 image_text_model=None,
                 image_text_model_type='siglip',
                 dpt_model=None,
                 processor=None,
                 depthanything_fn=None,
                 traj_len=10,
                 num_samples=500,
                 topk=50,
                 opt_steps=20,
                 var_scale=0.5,
                 smooth_weight=0.1,
                 visualize=False,
                 model_type="now",
                 generation_model=None,
                 plan_init_type="anchor"
                 ):

        self.traj_len = traj_len
        self.horizon = traj_len - 1  # 
        self.num_samples = num_samples
        self.topk = topk
        self.opt_steps = opt_steps
        self.var_scale = var_scale
        self.smooth_weight = smooth_weight
        self.visualize=visualize
        self.action_dim = 2
        self.goal=goal
        self.plan_init_type=plan_init_type

        self.model_type=model_type
        self.generation_model=generation_model
        logging.basicConfig(
            level=logging.INFO,  
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S' # 
        )
        if goal == "image":
            assert lpips_fn!=None
            self.lpips_fn=lpips_fn


        elif goal == "language":
            assert image_text_model!=None
            assert processor!=None
            self.image_text_model=image_text_model
            self.processor=processor
            self.image_text_model_type=image_text_model_type




    def _wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def get_trajs(self, actions):

        trajs =  np.zeros((actions.shape[0], actions.shape[1], 3))
        for t in range(1, trajs.shape[1]):
            current_pose = trajs[:, t-1, :]
            current_action = actions[:, t-1, :]
            dx = current_action[:, 0]
            dtheta = current_action[:, 1]
            
            global_dx = dx * np.cos(current_pose[:, 2])
            global_dy = dx * np.sin(current_pose[:, 2])
            new_theta = self._wrap_angle(current_pose[:, 2] + dtheta)
            
            trajs[:, t, 0] = current_pose[:, 0] + global_dx
            trajs[:, t, 1] = current_pose[:, 1] + global_dy
            trajs[:, t, 2] = new_theta


        return trajs



    def lpip_loss(self, obs_image, actions, x_target, bfloat_enable, num_cond, len_traj_pred, visualize):
        """
        obs_image:  [B,context_size,3,image_size,image_size]
        actions: [B,len_traj_pred+1,4]
        """
        latent_size = obs_image.shape[3] // 8
        B=actions.shape[0]

        # obs_image=obs_image.repeat((B,1,1,1,1))

        samples = self.generation_model.get_samples(obs_image, actions, num_cond=num_cond, len_traj_pred=len_traj_pred, \
            bfloat_enable=bfloat_enable, visualize=visualize)

        # samples = rearrange(samples, 'b c f2 h w -> b f2 c h w')

        with torch.no_grad():
            mse_loss = self.lpips_fn.forward(x_target, samples[:,-1,:,:,:]).squeeze(1,2,3).cpu().numpy()

        min_idx=np.argmin(mse_loss)

        return mse_loss

    def image_text_loss(self, obs_image, actions, goallan, bfloat_enable, num_cond, len_traj_pred, visualize):
        """
        obs_image:  [B,context_size,3,image_size,image_size]
        actions: [B,len_traj_pred+1,4]
        """
        latent_size = obs_image.shape[3] // 8
        B=actions.shape[0]

        # obs_image=obs_image.repeat((B,1,1,1,1))

        samples = self.generation_model.get_samples(obs_image, actions, num_cond=num_cond, len_traj_pred=len_traj_pred, \
            bfloat_enable=bfloat_enable, visualize=visualize)

        images_tensor = samples[:, -1, :, :, :] # 
        images_tensor = (images_tensor + 1.0) / 2.0
        images_tensor = (images_tensor * 255).clamp(0, 255).to(torch.uint8)
        images_np = images_tensor.permute(0, 2, 3, 1).cpu().numpy()
        pil_images = [Image.fromarray(img) for img in images_np]


        with torch.no_grad():
            

            if self.image_text_model_type in ['blip']:
            ### blip ###
                texts = [goallan] * len(pil_images)
                inputs = self.processor(
                    images=pil_images, 
                    text=texts, 
                    return_tensors="pt"
                ).to(self.image_text_model.device) #  CUDA
                outputs = self.image_text_model(**inputs)
                itm_logits = outputs.itm_score 
                probs = torch.nn.functional.softmax(itm_logits, dim=1)
                matching_scores = probs[:, 1]
            elif self.image_text_model_type in ['siglip', 'clip']:
            ### siglip / clip ###
                inputs = self.processor(text=[goallan], images=pil_images, padding="max_length", return_tensors="pt").to(self.image_text_model.device)
                outputs = self.image_text_model(**inputs)
                logits = outputs.logits_per_image
                # logging.info(f"{logits.shape}")
                matching_scores = logits.softmax(dim=0).squeeze()

        matching_scores = matching_scores.cpu().numpy()
        # logging.info(f"matching score{matching_scores}")
        mse_loss = 1 - matching_scores

        min_idx=np.argmin(mse_loss)

        return mse_loss


    def plan(self, obs_image, bfloat_enable, num_cond, len_traj_pred, goalimage=None, goallan=None, goalpos=None):

        initial_pose=(0,0,0)

        # print(self.plan_init_type)
        if self.plan_init_type=="anchor":
            var_scale=(0., 0.)
            v_vals = np.array([0.05, 0.1, 0.15, 0.2])                # dim=0
            w_vals = np.array([-0.15, -0.09, -0.05, -0.01, 0.01, 0.05, 0.09, 0.15])  # dim=1
            mu_list = np.array([[v, w] for v in v_vals for w in w_vals])     # (32, 2)

            mu = np.zeros((self.num_samples, self.horizon, self.action_dim))
            sigma = np.ones_like(mu) * var_scale 

            for i in range(self.num_samples):
                mu[i, :, 0] = mu_list[i][0]
                mu[i, :, 1] = mu_list[i][1]
        elif self.plan_init_type=="random":

            mu = np.zeros((self.horizon, self.action_dim))
            sigma = self.var_scale * np.ones((self.horizon, self.action_dim))

        best_traj = None
        best_loss = float('inf')

        sample_list=[]
        
        for step in range(self.opt_steps):
            tstart=time.time()

            samples = np.random.normal(
                loc=mu,
                scale=sigma,
                size=(self.num_samples, self.horizon, self.action_dim)
            )
            samples[samples[:,:,0]<0]=0
            actions = samples # single actions, with only distance and yaw delta

            trajs = self.get_trajs(actions)

            sample_list.append(trajs)
            visualize = False


            if self.goal == "image":
                mse_loss=self.lpip_loss(obs_image, torch.from_numpy(actions).float().to(obs_image.device), goalimage, bfloat_enable, num_cond, len_traj_pred, visualize)
            elif self.goal == "language":
                mse_loss=self.image_text_loss(obs_image, torch.from_numpy(actions).float().to(obs_image.device), goallan, bfloat_enable, num_cond, len_traj_pred, visualize)
     
            losses = mse_loss
        
            elite_idx = np.argsort(losses)[:self.topk]
            elite_samples = samples[elite_idx]
            elite_trajs = trajs[elite_idx]
            
            mu = elite_samples.mean(axis=0)
            sigma = elite_samples.std(axis=0) + 1e-6  # 
            
            current_best_loss = losses[elite_idx[0]]
            if current_best_loss < best_loss:
                best_loss = current_best_loss
                best_traj = elite_trajs[0]


        return best_traj, mu, sample_list