import os
import time
from util import statResults


nTimes = 5
expname = 'experiment23_stepdl'
mt = -1
curT = time.time()
task = 'condition' # or 'machine'
tempfile = f'temp-{curT}-{task}.csv'
outResultFile = f'Rs-{expname}-mt{mt}-{curT}-{task}.csv'
for _ in range(nTimes):
    os.system(f'CUDA_VISIBLE_DEVICES=4 python {expname}.py -mt {mt} --result_file {tempfile} --max_step 50000 --task {task}')
statResults(tempfile, outResultFile)
os.remove(tempfile)
