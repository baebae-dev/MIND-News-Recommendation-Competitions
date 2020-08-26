import os
import apex

gpus = '3,4,5,6'
num_gpus = len(gpus.strip().split(','))
# os.system("conda activate mind3")
# path = '~/.conda/envs/mind3/lib/python3.6/site-packages/'
python_path = '~/.conda/envs/mind3/bin/python3.6'
os.system(f"{python_path} -m torch.distributed.launch --nproc_per_node={num_gpus} main_origin.py --gpus {gpus}")
