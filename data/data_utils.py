# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import yaml
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


IMAGE_ASPECT_RATIO = (4 / 3)  # all images are centered cropped to a 4:3 aspect ratio in training

with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)


def get_action_torch(diffusion_output, action_stats):
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = torch.cumsum(ndeltas, dim=1)
    return actions.to(ndeltas)

def log_viz_single(dataset_name, obs_image, goal_image, preds, deltas, loss, min_idx, actions, action_stats, plan_iter=0, output_dir='plot.png'):
    '''
    Visualize a single instance
    actions is gt actions
    '''
    viz_obs_image = unnormalize(obs_image.detach().cpu())[-1] # take last img 
    viz_goal_image = unnormalize(goal_image.detach().cpu())
    deltas = deltas.detach().cpu()
    loss = loss.detach().cpu()
    actions = actions.detach().cpu()
    pred_actions = get_action_torch(deltas[:, :, :2], action_stats)
    plot_array = plot_images_and_actions(dataset_name, viz_obs_image, viz_goal_image, pred_actions, actions, min_idx, loss=loss)

    plt.imshow(plot_array)
    plt.axis('off')  # Hide axes for a cleaner image

    # Save the plot array as a PNG file locally
    plt.savefig(output_dir, format='png', dpi=300, bbox_inches='tight')

def plot_images_and_actions(dataset_name, curr_viz_obs_image, curr_viz_goal_image, curr_viz_pred_actions, curr_viz_actions, min_idx, loss):
    curr_viz_obs_image = curr_viz_obs_image.permute(1, 2, 0).cpu().numpy()
    curr_viz_goal_image = curr_viz_goal_image.permute(1, 2, 0).cpu().numpy()

    # scale back to metric space for plotting
    curr_viz_pred_actions = curr_viz_pred_actions * data_config[dataset_name]['metric_waypoint_spacing']
    curr_viz_actions = curr_viz_actions * data_config[dataset_name]['metric_waypoint_spacing']
    
    # Create the figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    # Plot condition image
    axs[0].imshow(curr_viz_obs_image)
    axs[0].set_title("Condition Image", fontsize=13)
    axs[0].axis("off")

    # Plot goal image
    axs[1].imshow(curr_viz_goal_image)
    axs[1].set_title("Goal Image", fontsize=13)
    axs[1].axis("off")

    colors = ['red', 'orange', 'cyan']
    for i in range(1, curr_viz_pred_actions.shape[0]):
        color = colors[(i - 1) % len(colors)]
        label = f"Sample {i} Min Loss" if i == min_idx.item() else f"{i}"

        if i != min_idx.item():
            axs[2].plot(-curr_viz_pred_actions[i, :, 1], curr_viz_pred_actions[i, :, 0], 
                        color=color, marker="o", markersize=5, label=label)
            axs[2].text(-curr_viz_pred_actions[i, -1, 1], 
                curr_viz_pred_actions[i, -1, 0], 
                round(loss[i].item(), 3), 
                color='black', 
                fontsize=10, 
                ha='left', va='bottom')  # Adjust position to avoid overlap

    # Highlight the minimum loss sample
    axs[2].plot(-curr_viz_pred_actions[min_idx.item(), :, 1], curr_viz_pred_actions[min_idx.item(), :, 0], 
                color='green', marker="o", markersize=5, label=f"{min_idx.item()}")
    axs[2].text(-curr_viz_pred_actions[min_idx.item(), -1, 1], 
        curr_viz_pred_actions[min_idx.item(), -1, 0], 
        round(loss[min_idx.item()].item(), 3), 
        color='black', 
        fontsize=10, 
        ha='left', va='bottom')  # Adjust position to avoid overlap

    # Plot ground truth actions
    axs[2].plot(-curr_viz_actions[:, 1], curr_viz_actions[:, 0], color='blue', marker="o", label="GT")

    # Set titles and labels with larger font size
    axs[2].set_title("   ", fontsize=13)
    axs[2].set_xlabel("X (m)", fontsize=11)
    axs[2].set_ylabel("Y (m)", fontsize=11)

    # Set equal aspect ratio and adjust axis limits
    axs[2].set_aspect('equal', adjustable='box')
    x_min, x_max = axs[2].get_xlim()
    y_min, y_max = axs[2].get_ylim()
    axis_range = max(x_max - x_min, y_max - y_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    axs[2].set_xlim(x_mid - axis_range, x_mid + axis_range)
    axs[2].set_ylim(y_mid - axis_range, y_mid + axis_range)

    axs[2].legend(loc='lower left', fontsize=10, frameon=True, bbox_to_anchor=(0, 0))
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_array = plot_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_array


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'].to(ndata) - stats['min'].to(ndata)) + stats['min'].to(ndata)
    return data

def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        # "image": ".jpg",## public dataset
        "image": ".png",## habitat dataset
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    
    return delta

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def calculate_delta_yaw(unnorm_actions):
    x = unnorm_actions[..., 0]
    y = unnorm_actions[..., 1]
    
    yaw = torch.atan2(y, x).unsqueeze(-1)
    delta_yaw = torch.cat((torch.zeros(yaw.shape[0], 1, yaw.shape[2]).to(yaw.device), yaw), dim=1)
    delta_yaw = delta_yaw[:, 1:, :] - delta_yaw[:, :-1, :]
    
    return delta_yaw


        
class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.ar)))
        else:
            img = TF.center_crop(img, (int(w / self.ar), w))
        return img

unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

def stitch_images_with_borders(obs_image, pred_image, line_thickness=10):
    """
    obs_image: tensor, shape (N1, 3, H, W)
    pred_image: tensor, shape (N2, 3, H, W)
    """
    assert obs_image.shape[1] == 3
    obs_image=obs_image.cpu()
    pred_image=pred_image.cpu()
    N, C, H, W = pred_image.shape

    images = []

    for i in range(obs_image.shape[0]):
        images.append(obs_image[i])
        if i != obs_image.shape[0] - 1:  
            black_line = torch.zeros(3, H, line_thickness)
            images.append(black_line)

    green_line = torch.zeros(3, H, line_thickness)
    green_line[1] = 1.0  
    images.append(green_line)

    for i in range(pred_image.shape[0]):
        images.append(pred_image[i])
        if i != pred_image.shape[0] - 1: 
            black_line = torch.zeros(3, H, line_thickness)
            images.append(black_line)

    stitched = torch.cat(images, dim=2)  

    return stitched  

def draw_traj_img(obs_image,pred_image,targetimage,actions):
    """
    obs_image: tensor, shape (f1, 3, H, W)
    pred_image: tensor, shape (f2, 3, H, W)
    targetimage: tensor, shape (f2, 3, H, W)
    """
    actions=actions.cpu().numpy()

    image_0=stitch_images_with_borders(obs_image,pred_image)
    image_0=(image_0.permute(1,2,0).cpu().numpy()*255).astype('uint8')

    fig, ax = plt.subplots(3,1,dpi=400)
    ax[0].imshow(image_0)
    ax[0].axis("off")  

    image=stitch_images_with_borders(obs_image,targetimage)
    image=(image.permute(1,2,0).cpu().numpy()*255).astype('uint8')

    ax[1].imshow(image)
    ax[1].axis("off")  

    ax[2].plot(actions[:,0], actions[:,1], 'b-o')
    ax[2].quiver(actions[:,0], actions[:,1], 
               actions[:,2], actions[:,3])
    ax[2].axis('equal')
    return fig    


def draw_traj_img_grid(obs_image,pred_image,actions):
    """
    obs_image: tensor, shape (1, f1, 3, H, W)
    pred_image: tensor, shape (B, f2, 3, H, W)
    targetimage: tensor, shape (B, f2, 3, H, W)
    """
    import matplotlib.gridspec as gridspec

    B=pred_image.shape[0]

    fig, ax = plt.subplots(dpi=500, figsize=(18, 3*B))

    gs = gridspec.GridSpec(B, 2, width_ratios=[3, 1])  

    ax = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(B)]

    actions=actions.cpu().numpy()

    for i in range(B):
        image_0=stitch_images_with_borders(obs_image[0],pred_image[i])
        image_0=(image_0.permute(1,2,0).cpu().numpy()*255).astype('uint8')

        ax[i][0].imshow(image_0)
        ax[i][0].axis("off")  


        ax[i][1].plot(actions[i,:,0], actions[i,:,1], 'b-o')
        ax[i][1].quiver(actions[i,:,0], actions[i,:,1], 
                actions[i,:,2], actions[i,:,3])
        ax[i][1].axis('equal')
    return fig    
