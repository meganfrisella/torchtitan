import subprocess
import sys
import os

local_rank = int(os.environ["LOCAL_RANK"])

args = sys.argv[1:]
args_string = " ".join(args)

print(f"Profiling local rank {local_rank}")
command = f"nsys profile --force-overwrite true -t cuda,nvtx -o out/torchtitan-rank{local_rank} python -m " + args_string

result = subprocess.run(command, shell=True)
sys.exit(result.returncode)