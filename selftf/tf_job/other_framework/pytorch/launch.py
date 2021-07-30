import subprocess
import sys
import os 
from torch.distributed.launch import __file__ as launch_file

cmd = sys.argv[1:]
idx = None
for arg in cmd:
    if "--task_index=" in arg:
        idx = arg.split("=")[1]
if idx == None:
    print("Cannot determine the task index. Exit.")
    exit(1)
cmd.insert(2, sys.argv[0].replace('launch.py', 'train.py'))
additional_args = f'{sys.executable} {launch_file} --nproc_per_node=1 --node_rank={idx} --master_port=1234'.split()
cmd = additional_args + cmd
proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=False,env=dict(os.environ))
while proc.poll() is None:
    line = proc.stdout.readline()
    line = line.strip()
    if line:
        print(line)
proc.wait()