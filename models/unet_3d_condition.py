# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================
# Modified from SeerVideoLDM
# Source: https://github.com/seervideodiffusion/SeerVideoLDM
# ==========================================================================
# NOTICE: The SeerVideoLDM repository does not provide an explicit license.
# Therefore, this modified module should be treated as RESTRICTED to 
# RESEARCH/NON-COMMERCIAL USE only, unless permission is granted by authors.
# ==========================================================================

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat, reduce

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils.import_utils import is_xformers_available
# import xformers
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    UNetMidBlock3DCrossAttn,
    DownBlock3D,
    UpBlock3D,
)
from .attention import (
    FeedForward,
    CrossAttention
)
MAX_LENGTH  = 1024
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class NOW(ModelMixin,ConfigMixin):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=512,
        attention_head_dim=8,
        action_dim=4,
        diffusion_type="diffusion", #"diffusion" or "shortcut"
    ):
        super().__init__()

        time_embed_dim = block_out_channels[0] * 4
        
        downsample_padding=1
        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.diffusion_type=diffusion_type
        assert self.diffusion_type in ["diffusion","shortcut"], "diffusion_type should be in [diffusion,shortcut]"
        if self.diffusion_type=="diffusion":
            # time
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        elif self.diffusion_type=="shortcut":
            # time
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim//2)

            #for shortcut "One-Step Diffusion via Shortcut Models"
            self.d_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            d_input_dim = block_out_channels[0]
            self.d_embedding = TimestepEmbedding(d_input_dim, time_embed_dim//2)


        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlock3D":
                down_block = CrossAttnDownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    causal = True,
                )
            else:
                down_block = DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    downsample_padding=downsample_padding,
                )
            
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            causal = True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock3D":
                up_block = CrossAttnUpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    causal = True,
                )
            else:
                up_block = UpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.action_embedding=nn.Sequential(
            nn.Linear(action_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,cross_attention_dim),
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        cond_sample: torch.FloatTensor,
        actions: torch.Tensor,
        cond_frame:int = 0,
        d: Union[torch.Tensor, float, int]=None,
    ) -> torch.FloatTensor:
        
        assert cond_sample.shape[2]==cond_frame, "cond frame num is conflicted"


        sample=torch.cat([cond_sample,sample],dim=2)

        # 0. get the action embedding
        context=self.action_embedding(actions)


        # 1. time
        timesteps = timestep
        # broadcast to batch dimension
        # timesteps = timesteps.broadcast_to(sample.shape[0])
        # d = d.broadcast_to(sample.shape[0])
        if self.diffusion_type=="diffusion":
            t_emb = self.time_proj(timesteps)
            emb = self.time_embedding(t_emb)

        elif self.diffusion_type=="shortcut":
            assert d!=None, "d should be input when using shortcut"
            t_emb = self.time_proj(timesteps)
            d_emb = self.d_proj(d)
            emb = torch.cat([self.time_embedding(t_emb),self.d_embedding(d_emb)],dim=1)

        # 2. pre-process
        sample = self.conv_in(sample)
        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):

            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=context, cond_frame = cond_frame
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=context, cond_frame = cond_frame)


        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context,
                    cond_frame = cond_frame, 
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = sample[:,:,cond_frame:,:,:]
        
        return output, None