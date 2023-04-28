import os
import time
from util import statResults


nTimes = 1 
mt = -1
curT = time.time()
task = 'condition' # or 'machine'
exp='exp_mobilefacenet_2s_sp_eval'
tempfile = f'temp-{exp}-mt{mt}-{curT}-{task}.csv'
outResultFile = f'Rs-{exp}-mt{mt}-{curT}-{task}.csv'
sp="True" #if True, use speed perturbed dataaset
device=1
max_step=40000
dur=2.0
batch_size=32
accumulate_grad_step=8
lr=0.001
for _ in range(nTimes):
    os.system(f'CUDA_VISIBLE_DEVICES={device} python experiment23_stepdl_w2v.py \
                -mt {mt} --result_file {tempfile} \
                --max_step {max_step} --task {task} -dur {dur} \
                -ep {exp} -bs {batch_size} -ac {accumulate_grad_step} \
                -sp {sp} -lr {lr}')
statResults(tempfile, outResultFile)
os.remove(tempfile)
