# panda_push_tqc_her

Training and evaluation code for TQC + HER on the Franka Panda push task, including domain randomization and physical and spatial generalization.

This repository contains the implementation used in the paper:

**From Generalization Cliffs to a Robust Domain: Physical and Spatial Boundaries in Robotic Pushing**

Follow-up measurements from October 2026 are in [`experiments/`](experiments/README.md).

## Acknowledgements

This project uses and modifies the open-source environments from:

- [panda_mujoco_gym](https://github.com/zichunxx/panda_mujoco_gym) by zichunxx

The vendored copy is `panda_mujoco_gym/`. `FrankaPushSparse-v0` keeps their task and adds the ranges used in the domain-randomization and generalization runs.

## Installation

Commands below are run from the repository root.

```bash
git clone https://github.com/Andy-Hyy/panda_push_tqc_her.git
cd panda_push_tqc_her
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

`requirements-delay-eval.txt` pins the June 2025 stack that loads the three baselines. Use that file when running `test/test_push_delay_generalization.py`.

## Training

```bash
python train/train_push_basic.py --seed 0 --batch_size 512 --n_sampled_goal 4
python train/train_push_multi-env.py --seed 0
python train/train_dr_multi-phy-elements_model.py
python train/train_dr_spatial.py
python train/train_fixed_spatial.py
python train/train_gamma.py --seed 0 --gamma 0.95
```

`train/train_dr_spatial.py` and `train/train_fixed_spatial.py` take no arguments. Each one trains seeds 42, 123, and 0. They are the June 2025 scripts. The October 2026 SpatialDR and Fixed Large comparison is:

```bash
python experiments/spatial_training_controlled_2026-10-02/scripts/train_controlled.py --method spatial_dr --seed 0
```

`--method` is `spatial_dr` or `fixed_large`.

`train/train_curriculum.py` needs `--stage` and `--seed`. `train/run_curriculum.sh` runs its three stages. `train/train_dr_multi-phy-elements_model-curriculum.py` is the other curriculum trainer and takes no arguments. `train/run_wrapper_gamma_experiments.sh` runs the remaining gamma sweeps.

## Evaluation

```bash
python test/test_push_basic.py --model-path path/to/final_model.zip
python test/test_push_basic_photo.py --model-path path/to/final_model.zip
python test/test_push_phycial_element_generalization.py --test-type physics --models-dir ./logs
python test/test_push_phycial_element_generalization.py --test-type multi --models-dir ./logs
python test/test_push_spatial_generalization.py --model-path path/to/final_model.zip
python test/test_dr_push.py --model-path path/to/final_model.zip
python test/test_push_delay_generalization.py --model-seed 0
python test/evaluate_robustness.py --model-path path/to/final_model.zip --param-name object_mass
python test/evaluate_robustness_extreme_joint.py --model-path path/to/final_model.zip
```

The physics launcher keeps the filename `test_push_phycial_element_generalization.py`. It calls `eval/worker_single_physics.py` and `eval/worker_multi_random.py`. `eval/test_multi_worker.py` is the coupled multi-parameter worker and takes `--model-path` and `--output-csv`.

`test/test_push_delay_generalization.py` reads the three June 2025 baselines from `models/`:

```text
models/batch512_her4_seed0_2025-06-17_13-50-58/final_model.zip
models/batch512_her4_seed42_2025-06-17_23-01-53/final_model.zip
models/batch512_her4_seed123_2025-06-18_23-21-02/final_model.zip
```

Those checkpoints stay outside this repository. The script writes into `experiments/physical_element/`. `requirements-delay-eval.txt` pins the stack that loads those baselines. `test/test_push_basic_photo.py` needs `pillow` and writes frames to `evaluation_frames/`.

## Layout

```text
panda_push_tqc_her/
├── train/
│   ├── train_push_basic.py
│   ├── train_push_multi-env.py
│   ├── train_dr_multi-phy-elements_model.py
│   ├── train_dr_spatial.py
│   ├── train_fixed_spatial.py
│   ├── train_curriculum.py
│   ├── train_dr_multi-phy-elements_model-curriculum.py
│   ├── train_gamma.py
│   ├── run_curriculum.sh
│   └── run_wrapper_gamma_experiments.sh
├── test/
│   ├── test_push_basic.py
│   ├── test_push_basic_photo.py
│   ├── test_push_delay_generalization.py
│   ├── test_push_phycial_element_generalization.py
│   ├── test_push_spatial_generalization.py
│   ├── test_dr_push.py
│   ├── evaluate_robustness.py
│   └── evaluate_robustness_extreme_joint.py
├── eval/
│   ├── worker_single_physics.py
│   ├── worker_multi_random.py
│   └── test_multi_worker.py
├── utils/
│   └── domain_rand.py
├── earlier/
├── panda_mujoco_gym/
├── experiments/
├── requirements.txt
├── requirements-delay-eval.txt
└── LICENSE
```

`earlier/` holds the domain-randomization drafts from before these entry points. Each folder keeps the `domain_rand.py` it imported.

## Reproducibility

The reported runs use seeds 0, 42, and 123. Training hyperparameters in the original scripts are batch size 512, HER k = 4, and gamma 0.95. The October protocols, checkpoint hashes, and success counts are in the reports under `experiments/`.

## License

MIT License
