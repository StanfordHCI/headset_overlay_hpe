import argparse
import os
import shutil
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

parser = argparse.ArgumentParser(description='Generate data for videopose')
parser.add_argument('--git_rev', type=str)
parser.add_argument('--resnet_model', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--additional_args', type=str)

args = parser.parse_args()

REPO_DIR = "/workspace/videopose3d"
PROJECT = "videopose3d"
OWNER = "jackiey"
EXPERIMENT = args.name
DATETIME = datetime.now().strftime('%Y%m%dT%H%M%S')
S3_OUTPUT_DIR = f"s3://geniehai/{OWNER}/video/{PROJECT}/processed_data/{EXPERIMENT}/{DATETIME}/"
RESNET_MODEL_S3 = args.resnet_model

DATA_DRIVE = "/data/"
DATA_RAW = os.path.join(DATA_DRIVE, "raw")
DATA_RAW_H36M = os.path.join(DATA_RAW, "h36m")
DATA_RAW_MPI = os.path.join(DATA_RAW, "mpi")


def checkout_repo(git_rev: str):
    os.system(f"cd {REPO_DIR} && git pull && git checkout {git_rev}")


def gen_3d_h36m():
    gen_process = subprocess.Popen(
        f"cd {REPO_DIR} &&"
        f"python3 data/prepare_data_h36m.py "
        f"--from-source-cdf {DATA_RAW_H36M} "
        f"{args.additional_args}",
        shell=True
    )
    gen_process.wait()


def sync_s3():
    os.system(f"aws s3 sync --no-progress data_* {S3_OUTPUT_DIR}")


if __name__ == '__main__':
    checkout_repo(args.git_rev)
    if args.mode == "3d_h36m":
        gen_3d_h36m()
        sync_s3()
