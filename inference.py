import os
import pickle
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Assuming this import exists in your environment
from sim_test.NavPolicy import NavPolicy 

def trajs_to_actions(trajs):
    B, T, _ = trajs.shape
    actions = np.zeros((B, T, 2))

    prev_x = np.zeros(B)
    prev_y = np.zeros(B)
    prev_theta = np.zeros(B)

    for t in range(T):
        curr_x = trajs[:, t, 0]
        curr_y = trajs[:, t, 1]
        curr_theta = trajs[:, t, 2]

        dx_world = curr_x - prev_x
        dy_world = curr_y - prev_y

        dx = dx_world * np.cos(prev_theta) + dy_world * np.sin(prev_theta)
        dtheta = ((curr_theta - prev_theta) + np.pi) % (2 * np.pi) - np.pi

        actions[:, t, 0] = dx
        actions[:, t, 1] = dtheta

        prev_x = curr_x
        prev_y = curr_y
        prev_theta = curr_theta

    return actions

# ==========================================
# === 1. Visualization Helpers ===
# ==========================================

def create_filmstrip(obs_tensor, samples_tensor):
    """
    Concatenates obs (frame 0) and samples (frames 1-N) into a horizontal strip.
    Args:
        obs_tensor: [1, 3, H, W] in range [-1, 1]
        samples_tensor: [1, T, 3, H, W] in range [-1, 1]
    Returns:
        np.array: (H, W_total, 3) ready for matplotlib
    """
    # 1. Unsqueeze obs to match samples dimensions: [1, 1, 3, H, W]
    if obs_tensor.dim() == 4:
        obs_tensor = obs_tensor.unsqueeze(1)
    
    # 2. Concatenate along time dimension: [1, T+1, 3, H, W]
    # Result shape: [1, 12, 3, 128, 128]
    full_seq = torch.cat([obs_tensor, samples_tensor], dim=1)
    
    # 3. Denormalize (-1, 1) -> (0, 1) and move to CPU
    full_seq = (full_seq * 0.5 + 0.5).clamp(0, 1).cpu()
    
    # 4. Convert to list of numpy images
    # Squeeze batch: [12, 3, 128, 128]
    seq_sq = full_seq.squeeze(0)
    num_frames, C, H, W = seq_sq.shape
    
    # Permute to HWC for plotting: [12, 128, 128, 3]
    seq_np = seq_sq.permute(0, 2, 3, 1).numpy()
    
    # 5. Horizontal Concatenation
    # List comprehension to grab each frame and stack horizontally
    filmstrip = np.concatenate([seq_np[i] for i in range(num_frames)], axis=1)
    
    return filmstrip

# ==========================================
# === 2. Comparison Visualization Function ===
# ==========================================
def visualize_comparison(obs_path, results, mode_name, save_path):
    """
    Visualizes goals, trajectories, and generated video predictions (filmstrips).
    
    Args:
        obs_path (str): Path to observation image.
        results (list): List of dicts. Each dict must have:
                        {'goal_input', 'goal_type', 'waypoints', 'filmstrip', 'label'}
        mode_name (str): Title suffix.
    """
    print(f"Generating comparison visualization -> {save_path}")

    # 1. Load Observation
    try:
        obs_img = Image.open(obs_path).convert('RGB')
    except Exception as e:
        print(f"Vis Error: Could not load obs image. {e}")
        return

    # 2. Setup Grid Layout
    # We need 5 Rows:
    # Row 0: Header Obs (Shared)
    # Row 1: Goal 1 | Traj 1
    # Row 2: Filmstrip 1 (Span all)
    # Row 3: Goal 2 | Traj 2
    # Row 4: Filmstrip 2 (Span all)
    
    fig = plt.figure(figsize=(16, 18)) # Increased height for filmstrips
    
    # height_ratios: Obs, (Goal/Traj), Strip, (Goal/Traj), Strip
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.6, 1, 0.5, 1, 0.5], width_ratios=[1, 1.2])

    # --- Plot 1: Observation (Top) ---
    ax_obs = fig.add_subplot(gs[0, :])
    ax_obs.imshow(obs_img)
    ax_obs.set_title(f"Current Observation (Shared)", fontsize=16, fontweight='bold', pad=10)
    ax_obs.axis('off')

    # --- Loop through the 2 results ---
    for i, res in enumerate(results):
        # Calculate Row Indices
        # i=0 -> data_row=1, strip_row=2
        # i=1 -> data_row=3, strip_row=4
        data_row = 1 + (i * 2)
        strip_row = data_row + 1
        
        # Extract Data
        goal_type = res['goal_type']
        goal_input = res['goal_input']
        waypoints = res['waypoints']
        filmstrip = res.get('filmstrip', None)
        label = res['label']

        # --- A. Plot Goal (Left Column) ---
        ax_goal = fig.add_subplot(gs[data_row, 0])
        ax_goal.set_title(f"Goal {i}: {label}", fontsize=13, fontweight='bold', color='darkblue')
        
        if goal_type == "image":
            try:
                g_img = Image.open(goal_input).convert('RGB')
                ax_goal.imshow(g_img)
                ax_goal.axis('off')
            except:
                ax_goal.text(0.5, 0.5, "Error Loading Image", ha='center')
        elif goal_type == "language":
            ax_goal.axis('off')
            wrapped_text = f"\"{goal_input}\""
            # Simple wrapping logic
            if len(wrapped_text) > 40:
                parts = [wrapped_text[k:k+40] for k in range(0, len(wrapped_text), 40)]
                wrapped_text = '\n'.join(parts)
            
            ax_goal.text(0.5, 0.5, wrapped_text, 
                         ha='center', va='center', fontsize=12, style='italic', 
                         bbox=dict(facecolor='#e6f2ff', edgecolor='blue', boxstyle='round,pad=1', alpha=0.3))

        # --- B. Plot Trajectory (Right Column) ---
        ax_traj = fig.add_subplot(gs[data_row, 1])
        ax_traj.set_title(f"Predicted Path for Goal {i}", fontsize=13, fontweight='bold')
        ax_traj.grid(True, linestyle=':', alpha=0.6)
        ax_traj.set_aspect('equal')

        if waypoints is not None and len(waypoints) > 0:
            pts = np.array(waypoints)
            if np.linalg.norm(pts[0]) > 0.05: 
                pts = np.concatenate([np.zeros((1, 2)), pts], axis=0)
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Set consistent scale
            ax_traj.set_xlim(-0.3, 2.0) 
            ax_traj.set_ylim(-1.0, 1.0)

            color = 'blue' if i == 0 else 'red'
            ax_traj.plot(x_vals, y_vals, color=color, linewidth=3, alpha=0.8, label=f'Traj {i}')
            ax_traj.scatter(x_vals, y_vals, c=color, marker='o', s=60)
            ax_traj.legend(loc='upper right', fontsize=10)
        else:
            ax_traj.text(0.5, 0.5, "No Path Found", ha='center', color='red')

        # --- C. Plot Generated Filmstrip (Span Both Columns, Below Traj) ---
        if filmstrip is not None:
            ax_strip = fig.add_subplot(gs[strip_row, :])
            ax_strip.imshow(filmstrip)
            ax_strip.set_title(f"Generated Future View (Frame 0 -> 11)", fontsize=10, style='italic')
            ax_strip.axis('off')
            
            # Add a small border
            for spine in ax_strip.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)

    plt.suptitle(f"Comparison: {mode_name}", fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to {save_path}")

# ==========================================
# === 3. Inference Logic (Updated) ===
# ==========================================

def get_inference_result(policy, obs_path, goal_input, goal_type):
    """
    Runs inference and returns a dictionary with waypoints and the generated filmstrip.
    """
    print(f"   -> Processing: {goal_type} | Goal: {str(goal_input)}...")

    # 1. Load Observation
    if not os.path.exists(obs_path):
        print(f"      Error: Obs file missing.")
        return {'waypoints': None, 'filmstrip': None}
    
    try:
        obs_image_raw = Image.open(obs_path).convert('RGB')
        obs_image = policy.transform(obs_image_raw).cuda()
        obs_image = obs_image * 2. - 1.
        obs_image = obs_image.unsqueeze(0).unsqueeze(0) # [1, 1, 3, H, W]
    except Exception as e:
        print(f"      Error loading obs: {e}")
        return {'waypoints': None, 'filmstrip': None}

    # 2. Handle Goal
    goal_data = None
    if goal_type == "image":
        if not os.path.exists(goal_input):
            print(f"      Error: Goal file missing.")
            return {'waypoints': None, 'filmstrip': None}
        try:
            goal_image_raw = Image.open(goal_input).convert('RGB')
            goal_image = policy.transform(goal_image_raw).cuda()
            goal_image = goal_image * 2. - 1.
            goal_data = goal_image
        except Exception as e:
            print(f"      Error loading goal img: {e}")
            return {'waypoints': None, 'filmstrip': None}
    elif goal_type == "language":
        goal_data = goal_input
    
    # 3. Plan (Get Waypoints)
    waypoints = None
    if goal_type == "image":
        waypoints, _, _ = policy.planner.plan(
            obs_image, bfloat_enable=True, num_cond=1, len_traj_pred=11, goalimage=goal_data
        )
    elif goal_type == "language":
        waypoints, _, _ = policy.planner.plan(
            obs_image, bfloat_enable=True, num_cond=1, len_traj_pred=11, goallan=goal_data
        )

    # 4. Generate Future Video (Samples)
    # Convert waypoints to actions
    actions = trajs_to_actions(waypoints[None, :, :])
    actions = torch.from_numpy(actions).to(obs_image.device)
    
    # Get Samples: Shape [1, 11, 3, 128, 128], range [-1, 1]
    samples = policy.generation_model.get_samples(
        obs_image, actions, num_cond=1, len_traj_pred=11, 
        bfloat_enable=True, visualize=True
    )
    
    # 5. Create Visualization Filmstrip
    # We strip the extra dim from obs_image for the helper if needed, but helper handles 4D/5D
    # obs_image is [1, 1, 3, 128, 128]
    filmstrip_img = create_filmstrip(obs_image, samples)

    # Process waypoints output to numpy
    pts_np = None
    if waypoints is not None:
        pts = waypoints[0] if (hasattr(waypoints, 'ndim') and waypoints.ndim == 3) else waypoints
        if isinstance(pts, torch.Tensor):
            pts_np = pts.detach().cpu().numpy()
        else:
            pts_np = pts

    return {
        'waypoints': pts_np,
        'filmstrip': filmstrip_img
    }

# ==========================================
# === 4. Main Configuration ===
# ==========================================

def main() -> None:
    # --- CONFIG ---
    OBS_PATH = "./infer_demo/current_obs.png" 
    
    # Scenario A: Image Goals
    GOAL_IMG_0 = "./infer_demo/goal_obs_0.png"
    GOAL_IMG_1 = "./infer_demo/goal_obs_1.png" 
    
    # Scenario B: Language Goals
    GOAL_TXT_0 = "Go to the wooden dining table with a rustic design."
    GOAL_TXT_1 = "Go to the hallway where the window with a view is on the left."
    
    SCENE_NAME = "ac26ZMwG7aT"
    MODEL_TYPE = "now"
    EXP_CONFIG = "config/config_shortcut_w_pretrain.yaml"
    MODEL_WEIGHTS = "./logs/shortcut_w_pretrain_20260114_161940/checkpoints/latest.pth.tar"

    DIST_MODEL_PATH = "/root/workspace/code/Navigation_worldmodel/worldmodelnav/models_dist/weights/EffoNav.pth"
    base_task_path = f"/root/workspace/code/Navigation_worldmodel/worldmodelnav/tasks/tasks-{SCENE_NAME}-image"

    print(f"Initializing NavPolicy...")

    # ==========================================
    # === TEST 1: Image Mode Comparison ===
    # ==========================================
    policy = NavPolicy(
        model_type=MODEL_TYPE, 
        goal_type="image", 
        task_path=base_task_path, 
        exp_eval=EXP_CONFIG,
        model_path=MODEL_WEIGHTS,
        plan_init_type="anchor", 
        dist_model_path=DIST_MODEL_PATH
    )
    print("\n=== Running Comparison 1: Image Goals ===")
    res_img_0 = get_inference_result(policy, OBS_PATH, GOAL_IMG_0, "image")
    res_img_1 = get_inference_result(policy, OBS_PATH, GOAL_IMG_1, "image")

    # Update results dict to unpack values
    results_image = [
        {
            'goal_input': GOAL_IMG_0, 'goal_type': 'image', 'label': 'Goal Image 0',
            'waypoints': res_img_0['waypoints'], 
            'filmstrip': res_img_0['filmstrip']
        },
        {
            'goal_input': GOAL_IMG_1, 'goal_type': 'image', 'label': 'Goal Image 1',
            'waypoints': res_img_1['waypoints'], 
            'filmstrip': res_img_1['filmstrip']
        }
    ]
    
    visualize_comparison(
        OBS_PATH, 
        results_image, 
        mode_name="Image Goals", 
        save_path="./infer_demo/waypoints_image_goals.png"
    )

    # ==========================================
    # === TEST 2: Language Mode Comparison ===
    # ==========================================
    policy = NavPolicy(
        model_type=MODEL_TYPE, 
        goal_type="language", 
        task_path=base_task_path, 
        exp_eval=EXP_CONFIG,
        model_path=MODEL_WEIGHTS,
        plan_init_type="anchor", 
        dist_model_path=DIST_MODEL_PATH
    )
    print("\n=== Running Comparison 2: Language Goals ===")
    res_txt_0 = get_inference_result(policy, OBS_PATH, GOAL_TXT_0, "language")
    res_txt_1 = get_inference_result(policy, OBS_PATH, GOAL_TXT_1, "language")

    results_text = [
        {
            'goal_input': GOAL_TXT_0, 'goal_type': 'language', 'label': 'Instruction 0',
            'waypoints': res_txt_0['waypoints'], 
            'filmstrip': res_txt_0['filmstrip']
        },
        {
            'goal_input': GOAL_TXT_1, 'goal_type': 'language', 'label': 'Instruction 1',
            'waypoints': res_txt_1['waypoints'], 
            'filmstrip': res_txt_1['filmstrip']
        }
    ]

    visualize_comparison(
        OBS_PATH, 
        results_text, 
        mode_name="Language Goals", 
        save_path="./infer_demo/waypoints_lang_goals.png"
    )

if __name__ == "__main__":
    main()