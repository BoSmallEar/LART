# input a folder and split its subfolders to running on different gpus
import os
import sys
import subprocess
import multiprocessing as mp
from concurrent import futures

GPUS=[0, 2, 3]

def run(dir, log_file):
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes
    gpu = GPUS[worker_id % len(GPUS)]
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python scripts/demo.py video.source={dir}"
    )

    print(f"LOGGING TO {log_file}")
    cmd = f"{cmd} > {log_file} 2>&1"
    print(cmd)
    subprocess.call(cmd, shell=True)

def main(root_folder):
    dir_list = os.listdir(root_folder)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=len(GPUS)) as exe:
        for dir in dir_list:
            log_file = f"{log_dir}/{dir}.log"
            dir_full=os.path.join(root_folder, dir)
            exe.submit(
                run,
                dir_full,
                log_file
            )


if __name__ == "__main__":
    root_folder = sys.argv[1]
    main(root_folder)