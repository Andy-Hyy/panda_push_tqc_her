# panda_push_tqc_her

Training and evaluation code for TQC + HER on Franka Panda push task, including domain randomization and physical/spatial generalization analysis.

This repository contains the implementation used in the paper:  
**"From Generalization Cliffs to a Robust Domain: Physical and Spatial Boundaries in Robotic Pushing"**

## Installation

```bash
git clone https://github.com/Andy-Hyy/panda_push_tqc_her.git
cd panda_push_tqc_her

pip install -r requirements.txt

# Install the custom panda_mujoco_gym environment (if you have a local modified version)
pip install -e ./panda_mujoco_gym
Quick Start
Training
Bash# Basic training (single environment)
python train/train_push_basic.py --seed 0 --batch_size 512 --n_sampled_goal 4

# Parallel training with multiple environments
python train/train_push_multi-env.py --seed 0

# Domain Randomization training (multi physical parameters)
python train/train_dr_multi-phy-elements_model.py
Evaluation
Bash# Basic model evaluation
python eval/test_push_basic.py --model-path path/to/final_model.zip

# Physical parameter generalization test
python eval/tests_push_phycial_element_generalization.py --test-type physics --models-dir ./logs

# Multi-parameter joint perturbation test
python eval/tests_push_phycial_element_generalization.py --test-type multi --models-dir ./logs

# Spatial generalization test
python eval/test_push_spatial_generalization.py --model-path path/to/final_model.zip
Project Structure
textpanda_push_tqc_her/
├── train/                          # Training scripts
│   ├── train_push_basic.py
│   ├── train_push_multi-env.py
│   └── train_dr_multi-phy-elements_model.py
├── eval/                           # Evaluation scripts
│   ├── test_push_basic.py
│   ├── test_push_spatial_generalization.py
│   └── tests_push_phycial_element_generalization.py
├── utils/                          # Helper modules (DomainRandomizationWrapper, etc.)
├── requirements.txt
└── README.md
Reproducibility

All experiments use seeds: [0, 42, 123]
Hyperparameters follow the paper settings (batch size = 512, HER k = 4, γ = 0.95, etc.)
Domain randomization ranges and evaluation protocols are consistent with the paper

License
MIT License

Acknowledgements

This project uses and modifies the open-source environments from:

- [panda_mujoco_gym](https://github.com/zichunxx/panda_mujoco_gym) by zichunxx  
  (Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator)

The `FrankaPushSparse-v0` environment is based on their implementation, with modifications made to support our domain randomization and generalization experiments.
