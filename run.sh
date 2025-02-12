TRAIN_FLAGS="--iterations 200000 --anneal_lr True --batch_size 8 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"


mpiexec -n 1 python scripts/classifier_train.py --data_dir /home/saranshvashistha/workspace/class_conditioned/guided-diffusion/datasets/CottonWeedDiff_train $TRAIN_FLAGS $CLASSIFIER_FLAGS

sudo apt install lam-runtime       # version 7.1.4-7, or
sudo apt install mpich             # version 4.0-3
sudo apt install openmpi-bin       # version 4.1.2-2ubuntu1
sudo apt install slurm-wlm-torque  # version 21.08.5-2ubuntu1



Hi! Made a little edit in setup.py such that the package is available globally when installed.

Instead of doing this in the diffusion notebooks:

!git clone https://github.com/crowsonkb/guided-diffusion
pip install -e ./guided-diffusion
sys.path.append('./guided-diffusion')
You can do without manually adding the folder to path, simplifying to just this:

!pip install git+https://github.com/crowsonkb/guided-diffusion



sys.path.append("./guided-diffusion")
