"""
Collect navigation trajectories in Habitat and save in format compatible with
NOW/data/dataset.py and data_utils.py (per-trajectory folders, traj_data.pkl, traj_names.txt).
"""
import argparse
import os
import pickle
import shutil

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_to_angle_axis


# ================= Configurable parameters =================
SCENE_ID = "2t7WUuJeko7"
DATA_ROOT = "/root/workspace/d6mlljsp420c73eh0hcg/data/mp3d"
OUTPUT_DIR = None

IMAGE_H = 256
IMAGE_W = 256
CAMERA_HEIGHT = 1.

NUM_TRAJECTORIES = 500
GOAL_RADIUS = 0.25
MIN_TRAJ_FRAMES = 2
MAX_SAMPLE_ATTEMPTS = 50
# ============================================


def get_yaw(rotation_quat):
    """
    Extract yaw (heading, rotation around Y axis) from Habitat quaternion.
    Uses angle-axis: heading = theta * w[1].
    """
    theta, w = quat_to_angle_axis(rotation_quat)
    heading = theta * w[1]
    return float(heading)


def build_scene_path(data_root: str, scene_id: str) -> str:
    """data_root/scene_id/scene_id.glb"""
    return os.path.join(data_root, scene_id, f"{scene_id}.glb")


def _sample_start_goal(pathfinder, min_euclidean=1.0, min_geodesic_steps=2, max_attempts=50, fixed_start=None):
    """
    Sample start and goal points with min euclidean and geodesic requirements.
    fixed_start: if provided, start point fixed to this position; otherwise randomly sampled.
    Returns (start_pos, goal_pos) or None.
    """
    for _ in range(max_attempts):
        if fixed_start is not None:
            start_pos = np.array(fixed_start, dtype=np.float32)
        else:
            start_pos = pathfinder.get_random_navigable_point()
        goal_pos = pathfinder.get_random_navigable_point()
        if np.linalg.norm(np.array(goal_pos) - np.array(start_pos)) < min_euclidean:
            continue
        path = habitat_sim.nav.ShortestPath()
        path.requested_start = np.array(start_pos, dtype=np.float32)
        path.requested_end = np.array(goal_pos, dtype=np.float32)
        found = pathfinder.find_path(path)
        if not found or path.points is None or len(path.points) < min_geodesic_steps:
            continue
        return start_pos, goal_pos
    return None


def collect_one_trajectory(sim, agent, follower, output_traj_dir, goal_radius, min_frames=2, max_sample_attempts=50, start_from_previous=False):
    """
    Move along shortest path, save images and poses.
    start_from_previous: if True, start from current agent position, only resample goal; if False, randomly sample both start and goal.
    Returns (frames_saved, reached_goal, valid). If valid is False, no pkl is written and the directory is deleted.
    The save format is:
    - images: 0.png, 1.png, ...
    - traj_data.pkl: (T, 3) numpy
    """
    pathfinder = sim.pathfinder
    fixed_start = None
    if start_from_previous:
        fixed_start = agent.get_state().position
    pair = _sample_start_goal(
        pathfinder,
        min_euclidean=1.0,
        min_geodesic_steps=min_frames + 1,
        max_attempts=max_sample_attempts,
        fixed_start=fixed_start,
    )
    if pair is None:
        return 0, False, False
    start_pos, goal_pos = pair

    state = agent.get_state()
    state.position = start_pos
    agent.set_state(state)

    os.makedirs(output_traj_dir, exist_ok=True)
    positions = []
    step_idx = 0
    max_steps = 2000

    while step_idx < max_steps:
        obs = sim.get_sensor_observations()
        current_state = agent.get_state()
        pos = current_state.position
        yaw = get_yaw(current_state.rotation)

        img_rgba = obs["color_sensor"]
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        img_path = os.path.join(output_traj_dir, f"{step_idx}.png")
        cv2.imwrite(img_path, img_bgr)

        positions.append([float(pos[0]), float(pos[2]), float(yaw)])

        dist_to_goal = np.linalg.norm(np.array(pos) - np.array(goal_pos))
        if dist_to_goal <= goal_radius:
            break

        try:
            action = follower.next_action_along(goal_pos)
            if action is None:
                break
            sim.step(action)
            step_idx += 1
        except habitat_sim.errors.GreedyFollowerError:
            break

    n_frames = len(positions)
    reached = step_idx < max_steps and np.linalg.norm(np.array(agent.get_state().position) - np.array(goal_pos)) <= goal_radius

    if n_frames < min_frames:
        if os.path.isdir(output_traj_dir):
            shutil.rmtree(output_traj_dir)
        return n_frames, reached, False

    traj_array = np.array(positions, dtype=np.float64)
    with open(os.path.join(output_traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_array, f)
    return n_frames, reached, True


def main():
    parser = argparse.ArgumentParser(description="Collect Habitat trajectories for NOW dataset.")
    parser.add_argument("--scene", type=str, default=SCENE_ID, help="Scene id (e.g. 2t7WUuJeko7)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT, help="MP3D data root")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: habitat/{scene_id} under same parent as data-root)")
    parser.add_argument("--image-h", type=int, default=IMAGE_H, help="Image height")
    parser.add_argument("--image-w", type=int, default=IMAGE_W, help="Image width")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT, help="Camera height above agent (m)")
    parser.add_argument("--num-traj", type=int, default=NUM_TRAJECTORIES, help="Number of trajectories to collect")
    parser.add_argument("--goal-radius", type=float, default=GOAL_RADIUS, help="Goal reached radius (m)")
    parser.add_argument("--min-frames", type=int, default=MIN_TRAJ_FRAMES, help="Min frames per trajectory (discard if fewer)")
    parser.add_argument("--max-sample-attempts", type=int, default=MAX_SAMPLE_ATTEMPTS, help="Max attempts to sample start/goal per trajectory")
    args = parser.parse_args()

    scene_path = build_scene_path(args.data_root, args.scene)
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    output_dir = args.output_dir
    if output_dir is None or output_dir == "":
        parent = os.path.dirname(os.path.abspath(args.data_root))
        output_dir = os.path.join(parent, "habitat", args.scene)
    os.makedirs(output_dir, exist_ok=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [args.image_h, args.image_w]
    rgb_sensor_spec.position = [0.0, args.camera_height, 0.0]
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    follower = habitat_sim.nav.GreedyGeodesicFollower(
        sim.pathfinder, sim.agents[0], goal_radius=args.goal_radius
    )
    agent = sim.initialize_agent(0)

    traj_names = []
    i = 0
    attempts = 0
    max_total_attempts = args.num_traj * (args.max_sample_attempts + 5)
    while len(traj_names) < args.num_traj and attempts < max_total_attempts:
        attempts += 1
        traj_name = f"traj_{len(traj_names):04d}"
        traj_dir = os.path.join(output_dir, traj_name)
        start_from_previous = len(traj_names) > 0
        n_frames, reached, valid = collect_one_trajectory(
            sim, agent, follower, traj_dir, args.goal_radius,
            min_frames=args.min_frames, max_sample_attempts=args.max_sample_attempts,
            start_from_previous=start_from_previous,
        )
        if not valid:
            if n_frames == 0:
                print(f"  skip (no valid start/goal path)")
            else:
                print(f"  skip (only {n_frames} frames, need >={args.min_frames})")
            continue
        traj_names.append(traj_name)
        status = "reached" if reached else "stopped"
        print(f"[{len(traj_names)}/{args.num_traj}] {traj_name} frames={n_frames} {status}")

    traj_names_path = os.path.join(output_dir, "traj_names.txt")
    with open(traj_names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(traj_names) + "\n")

    sim.close()
    print(f"Done. Collected {len(traj_names)} trajectories. traj_names.txt -> {traj_names_path}")


if __name__ == "__main__":
    main()
