# Training and Inference

## 1. Data Preparation

### 1.1 Public Datasets
For acquiring and processing public datasets, please follow [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling).

### 1.2 Dataset Structure
Your dataset must follow the directory structure below. If you are collecting a custom dataset, please organize it accordingly:

```text
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── ...
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
        ├── 0.jpg
        ├── ...
        └── traj_data.pkl
```

### 1.3 Data Splitting
Use `data_split.py` to split your data into training and testing sets.

**Split Training Data:**
```bash
python data_split.py -i <path_to_train_data> -d <train_dataset_name>
```

**Split Test Data:**
```bash
python data_split.py -i <path_to_test_data> -d <test_dataset_name> -s 0
```

---

## 2. Training

Train the model using the provided configuration file:

```bash
python train.py --config config/config_shortcut_w_pretrain.yaml
```

---

## 3. Generation Performance Evaluation

Evaluation involves three steps: preparing ground truth frames, generating future frames, and calculating metrics (LPIPS, DreamSim, FID).

**Step 1: Prepare Ground Truth Frames**
```bash
python isolated_infer.py --exp logs/<run_name> --ckp latest --datasets <test_dataset_name> --gt 1
```

**Step 2: Generate Future Frames**
```bash
python isolated_infer.py --exp logs/<run_name> --ckp latest --datasets <test_dataset_name> --gt 0
```

**Step 3: Calculate Metrics**
```bash
python isolated_eval.py --gt_dir output/gt --exp_dir output/<run_name>_latest --datasets <test_dataset_name>
```

---

## 4. Inference

Perform waypoints prediction using the trained World Model.

### 4.1 Prerequisites
Download the [Distance Model Weights](https://drive.google.com/file/d/1Xf7FQNkdId0gSNfKy3TsMsytzUHpMc4A/view?usp=drive_link) before running inference into `models_dist/weights`.

### 4.2 Run Inference
We provide an intuitive inference script for testing:

```bash
python inference.py
```