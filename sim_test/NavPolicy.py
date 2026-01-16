import math
import multiprocessing
import os
import random
import time
from enum import Enum
import numpy as np
from PIL import Image
import logging
import yaml, json, torch, lpips, os
from torchvision import transforms as T
from models_dist.EffoNav.EffoNav import EffoNav
from models_dist.EffoNav.utilities import DinoV2ExtractFeatures
import pickle
from transformers import (
    AutoProcessor, AutoModel, 
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForImageTextRetrieval,
    SiglipProcessor, SiglipModel
)


_barrier = None

import torchvision.transforms.functional as TF
import torchvision.transforms as T
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training
IMAGE_RESIZE_SIZE=[85, 64]
def transform_images(
    img, image_resize_size = IMAGE_RESIZE_SIZE, aspect_ratio = IMAGE_ASPECT_RATIO
):
    h, w = img.shape[2], img.shape[3]

    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))

    # img = img.resize(image_resize_size)

    transform=T.Compose([
            T.Resize([IMAGE_RESIZE_SIZE[1],IMAGE_RESIZE_SIZE[0]]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transf_img = transform(img)
    return  transf_img

class NavPolicy:
    def __init__(self, task_path,\
                exp_eval,\
                model_path,\
                dist_model_path,\
                model_type="now",\
                goal_type="image",\
                plan_init_type="anchor",
                image_text_model_type="siglip"
        ):
        self.device = torch.device("cuda:0")

        self.task_path=task_path
        self.exp_eval=exp_eval
        self.model_path=model_path
        self.dist_model_path=dist_model_path
        self.model_type=model_type
        self.goal_type=goal_type
        self.plan_init_type=plan_init_type
        self.image_text_model_type=image_text_model_type

        self.radius = 3
        self.goal_stride = 2

        self.init_model()
        self.init_planner()

        folders = [f for f in os.listdir(self.task_path)
                if os.path.isdir(os.path.join(self.task_path, f)) and f.startswith("goal_nodes_")]

        folders_sorted = sorted(folders, key=lambda x: int(x.split("_")[-1]))
        self.goal_folders = folders_sorted
        self.task_id = 0

        self.start = False


    def init_model(self):
        if "now" in self.model_type:
            from sim_test.generation_models import get_generation_model
            self.generation_model = get_generation_model("now", self.exp_eval, self.model_path)

        self.transform = T.Compose([
            T.Resize((128,128)), #T.CenterCrop(args.resolution),
            T.ToTensor(),
        ])
        print("loaded models done !")


    def init_planner(self):
        visualize = False
        samples=32
        topk=10
        opt_steps=5


        dist_model = EffoNav(
            context_size=5,
            len_traj_pred=5,
            learn_angle=True,
            obs_encoder= "efficientnet-b0",
            obs_encoding_size=512,
            late_fusion=False,
            mha_num_attention_heads=4,
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4,
        ).to(self.device)


        dist_model.load_state_dict(torch.load(self.dist_model_path), strict=True)

        DINO=DinoV2ExtractFeatures(dino_model="dinov2_vits14", layer=11, facet='value',device=self.device)

        print("loaded EffoNAV and DINO weigthts !")

        from sim_test.CEM_planner import CEMTrajectoryPlanner
        if self.goal_type=="image":
            lpips_fn=lpips.LPIPS(net='alex', ).to(self.device)
        
            self.planner = CEMTrajectoryPlanner(
                goal="image",
                traj_len=12,
                num_samples=samples,
                topk=topk,
                opt_steps=opt_steps,
                var_scale=(0.3, 0.4),
                lpips_fn=lpips_fn,
                visualize=visualize,
                model_type=self.model_type,
                generation_model=self.generation_model,
                plan_init_type=self.plan_init_type
            )
        elif self.goal_type=="language":
            if self.image_text_model_type == "blip":
                ###blip###
                local_path = "/root/workspace/code/Navigation_worldmodel/worldmodelnav/sim_test/weights/blip-itm-base-coco"
                processor = BlipProcessor.from_pretrained(local_path)
                image_text_model = BlipForImageTextRetrieval.from_pretrained(local_path).to(self.device)

            elif self.image_text_model_type == "clip":
                ###clip###
                local_path = "/root/workspace/code/Navigation_worldmodel/worldmodelnav/sim_test/weights/clip-vit-base-patch32"
                processor = AutoProcessor.from_pretrained(local_path)
                image_text_model = AutoModel.from_pretrained(local_path).to(self.device)

            elif self.image_text_model_type == "siglip":
                ###siglip###
                local_path = "/root/workspace/code/Navigation_worldmodel/worldmodelnav/sim_test/weights/siglip-base-patch16-224"
                processor = AutoProcessor.from_pretrained(local_path)
                image_text_model = AutoModel.from_pretrained(local_path).to(self.device)


            self.planner = CEMTrajectoryPlanner(
                goal="language",
                traj_len=12,
                num_samples=samples,
                topk=topk,
                opt_steps=opt_steps,
                var_scale=(0.3, 0.4),
                image_text_model=image_text_model,
                image_text_model_type=self.image_text_model_type,
                processor=processor,
                visualize=visualize,
                model_type=self.model_type,
                generation_model=self.generation_model,
                plan_init_type=self.plan_init_type
            )

        self.dist_model = dist_model
        self.DINO = DINO
        self.dist_model.eval()

        print("loaded planner done !")



    def model_compute_wm(self, obs):

        if obs["start_new_episode"]:
            
            goal_node_path = os.path.join(self.task_path, obs["goal_nodes_path"])
            
            if self.goal_type=="language":
                with open(os.path.join(goal_node_path,"langoal.txt"), 'r', encoding='utf-8') as f:
                    self.langoal = f.read() 
                self.reached_goal = False

            elif self.goal_type=="image":
                png_count = len([f for f in os.listdir(goal_node_path) if f.lower().endswith(".png")])
                self.topomap=[]
                goals = [ os.path.join(goal_node_path,f"{i}.png") for i in sorted(set(list(range(0, png_count, 5)) + [png_count-1]))]

                for goal_path in goals:
                    self.topomap.append(Image.open(goal_path).convert('RGB'))

                self.closest_node = 0
                self.goal_node = len(self.topomap)-1
                self.reached_goal = False
                print("refresh the topomap and nav processing: ",goal_node_path, flush=True)
            elif self.goal_type=="point":
                self.reached_goal = False
                pass

            result={
                "action": np.zeros(2),
                "server_ready": True
            }
            self.start = True
            return result

        elif not self.start:
            result={
                "action": np.zeros(2),
                "server_ready": False
            }
            return result

        
        obs_image = Image.fromarray((obs["observation/image"]))
        obs_image = self.transform(obs_image).to(self.device)
        obs_image = obs_image*2. - 1.
        obs_image=obs_image.unsqueeze(0).unsqueeze(0)
        
        goalpos = obs["observation/goalpos"]
        total_frames = obs["observation/total_frames"]

        t0=time.time()
        if self.goal_type=="image":
    ####localization and subgoal dicision
            with torch.no_grad():
                start = max(self.closest_node - self.radius, 0)
                end = min(self.closest_node + self.radius + 1, self.goal_node)
                distances = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(self.topomap[start: end + 1]):
                    transf_obs_img = transform_images(obs_image[0]*0.5 + 0.5).repeat(1, 6, 1, 1)
                    goal_data = transform_images(T.ToTensor()(sg_img).unsqueeze(0))
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)

                distances, _ = self.dist_model( batch_obs_imgs.float(), batch_goal_data.float(), self.DINO) 
                min_dist_idx = np.argmin(distances.cpu().numpy())

                # chose subgoal and output waypoints
                # if distances[min_dist_idx] > 3:
                self.closest_node = start + min_dist_idx
                goal_id = min(self.closest_node + self.goal_stride, self.goal_node)

                self.reached_goal = self.closest_node == self.goal_node

                x_target_pil = self.topomap[goal_id]
                x_target = self.transform(x_target_pil).to(self.device)
                x_target=x_target*2.-1.
    ######################################

            if "now" in self.model_type:
                waypoints, best_actions, sample_list = self.planner.plan(obs_image, bfloat_enable=True, num_cond=1, len_traj_pred=11, goalimage=x_target)

        elif self.goal_type=="language":
            if "now" in self.model_type:
                waypoints, best_actions, sample_list = self.planner.plan(obs_image, bfloat_enable=True, num_cond=1, len_traj_pred=11, goallan=self.langoal)


        dx, dy = waypoints[4][0], waypoints[4][1]

        MAX_V, MAX_W = 10, 20
        DT = 1/30
        EPS = 1e-8
        
        if np.abs(dx) < EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*DT)
        else:
            v = dx / DT / 3
            w = np.arctan(dy/dx) / DT * 1.5

        v = np.clip(v, 0, MAX_V).item()
        w = np.clip(w, -MAX_W, MAX_W).item()

        if self.reached_goal:
            v,w = 0,0
            print("Reached goal! Stopping...")

        print(f"v: {v}, w: {w}, latency: {time.time()-t0}", flush=True)

        result={
            "action": np.array([v,w]),
            "server_ready": False
            }
        return result


    def model_compute_v_w(self, obs):
        return self.model_compute_wm(obs)