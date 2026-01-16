import torch
from einops import rearrange

@torch.no_grad()
def model_forward_wrapper(all_models, obs_image, actions, num_cond, len_traj_pred, latent_size, device, bfloat_enable, progress=False):
    model, diffusion, vae = all_models
    obs_image = obs_image.to(device)
    actions = actions.to(device)

    with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
        B = obs_image.shape[0]
        obs_image = obs_image.flatten(0,1)

        latents_obs = vae.encode(obs_image).latent_dist.sample().mul_(0.18215).unflatten(0, (B, num_cond))
        latents_obs = latents_obs.permute(0,2,1,3,4)# (b c f h w)

        z = torch.randn(B, 4, len_traj_pred, latent_size, latent_size, device=device)
        model_kwargs = dict(cond_sample=latents_obs,actions=actions,cond_frame=num_cond)

        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        ) # (b,c,f2,h,w)
        samples = rearrange(samples, 'b c f2 h w -> (b f2) c h w')
        samples = vae.decode(samples / 0.18215).sample
        samples = rearrange(samples, '(b f2) c h w -> b c f2 h w', f2=len_traj_pred)

        return torch.clip(samples, -1., 1.)