import subprocess
import sys
import os

local_rank = int(os.environ["LOCAL_RANK"])

args = sys.argv[1:]
args_string = " ".join(args)

print(f"Profile local rank {local_rank} only")
if local_rank == 0:
    command = "nsys profile -t cuda,nvtx -o test_run python -m " + args_string
else:
    command = "python -m " + args_string

result = subprocess.run(command, shell=True)
sys.exit(result.returncode)