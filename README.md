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

## Training

```bash
python train/train_push_basic.py --seed 0 --batch_size 512 --n_sampled_goal 4
python train/train_push_multi-env.py --seed 0
python train/train_dr_multi-phy-elements_model.py
```

## Evaluation

```bash
python test/test_push_basic.py --model-path path/to/final_model.zip
python test/test_push_phycial_element_generalization.py --test-type physics --models-dir ./logs
python test/test_push_phycial_element_generalization.py --test-type multi --models-dir ./logs
python test/test_push_spatial_generalization.py --model-path path/to/final_model.zip
```

The physics launcher keeps the filename `test_push_phycial_element_generalization.py`. It calls the workers in `eval/`.

## Layout

```text
panda_push_tqc_her/
├── train/
│   ├── train_push_basic.py
│   ├── train_push_multi-env.py
│   └── train_dr_multi-phy-elements_model.py
├── test/
├── eval/
│   ├── worker_single_physics.py
│   └── worker_multi_random.py
├── utils/
│   └── domain_rand.py
├── panda_mujoco_gym/
├── experiments/
├── requirements.txt
└── LICENSE
```

## Reproducibility

The reported runs use seeds 0, 42, and 123. Training hyperparameters in the original scripts are batch size 512, HER k = 4, and gamma 0.95. The October protocols, checkpoint hashes, and success counts are in the reports under `experiments/`.

## License

MIT License
