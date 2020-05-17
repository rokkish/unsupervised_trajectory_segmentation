"""
set default of my_args.py
"""
# my_args.py
animal = "bird"
epoch = 2**5
epoch_all = 2**3
alpha = 10
lambda_p = 0.01
tau = 100
start = 0
end = 5
net = "segnet"
lr = 0.05
momentum = 0.9
dir_name = ""
K = 10
label_dir = "result/paper_xmodify_bird_xyt_sec_arg_myloss_a0.1_p0.01_tau10000.0"
GPU_ID = 0

# set_hypara.py
mod_dim1 = 64
mod_dim2 = 32
MIN_LABEL_NUM = 4  # if the label number small than it, break loop
window = 20