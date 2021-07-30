import subprocess
import sys
import os 

cmd = sys.argv[1:]
idx = None
for arg in cmd:
    if "--task_index=" in arg:
        idx = int(arg.split("=")[1])
if idx == 0:
    cmd.insert(0, "horovodrun")
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=False,env=dict(os.environ))
    while proc.poll() is None:
        line = proc.stdout.readline()
        line = line.strip()
        if line:
            print(line)
    proc.wait()