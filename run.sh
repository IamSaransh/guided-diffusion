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



# Diffusion Model trianaing
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 2 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 256  --resume_checkpoint /home/saranshvashistha/workspace/class_conditioned/guided-diffusion/models/0000.pt --num_head_channels 64 --attention_resolutions 32,16,8"
OPENAI_LOGDIR=/home/saranshvashistha/workspace/class_conditioned/guided-diffusion/logs

mpiexec -n 1 python scripts/image_train.py --data_dir /home/saranshvashistha/workspace/class_conditioned/guided-diffusion/datasets/CottonWeedDiff_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --no_cuda

[-h] [--data_dir DATA_DIR] [--schedule_sampler SCHEDULE_SAMPLER] [--lr LR] [--weight_decay WEIGHT_DECAY] [--lr_anneal_steps LR_ANNEAL_STEPS] [--batch_size BATCH_SIZE] [--microbatch MICROBATCH]
                      [--ema_rate EMA_RATE] [--log_interval LOG_INTERVAL] [--save_interval SAVE_INTERVAL] [--resume_checkpoint RESUME_CHECKPOINT] [--use_fp16 USE_FP16] [--fp16_scale_growth FP16_SCALE_GROWTH]
                      [--image_size IMAGE_SIZE] [--num_channels NUM_CHANNELS] [--num_res_blocks NUM_RES_BLOCKS] [--num_heads NUM_HEADS] [--num_heads_upsample NUM_HEADS_UPSAMPLE] [--num_head_channels NUM_HEAD_CHANNELS]
                      [--attention_resolutions ATTENTION_RESOLUTIONS] [--channel_mult CHANNEL_MULT] [--dropout DROPOUT] [--class_cond CLASS_COND] [--use_checkpoint USE_CHECKPOINT]
                      [--use_scale_shift_norm USE_SCALE_SHIFT_NORM] [--resblock_updown RESBLOCK_UPDOWN] [--use_new_attention_order USE_NEW_ATTENTION_ORDER] [--learn_sigma LEARN_SIGMA] [--diffusion_steps DIFFUSION_STEPS]
                      [--noise_schedule NOISE_SCHEDULE] [--timestep_respacing TIMESTEP_RESPACING] [--use_kl USE_KL] [--predict_xstart PREDICT_XSTART] [--rescale_timesteps RESCALE_TIMESTEPS]
                      [--rescale_learned_sigmas RESCALE_LEARNED_SIGMAS]