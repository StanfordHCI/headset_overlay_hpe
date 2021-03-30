import argparse
import os
import shutil
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

parser = argparse.ArgumentParser(description='Generate data for videopose')
parser.add_argument('--git_rev', type=str)
parser.add_argument('--update_data', type=str)
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
DATA_RAW_H36M_S3 = "s3://geniehai/jackiey/videopose3d/raw/h36m.tar.bz2"
DATA_RAW_MPI_S3 = "s3://geniehai/jackiey/videopose3d/raw/mpi.tar.bz2"


def checkout_repo(git_rev: str):
    os.system(f"cd {REPO_DIR} && git pull && git checkout {git_rev}")

def pull_data():
    if not os.path.exists(DATA_DIR) or args.update_data == 'true':
        if args.update_data == 'true':
            shutil.rmtree(DATA_DIR)
        print("pulling data")
        os.system(f"aws s3 cp {DATA_S3} {DATA_DRIVE}")
        os.system(f"cd {DATA_DRIVE} && pbzip2 -dc data.tar.bz2 | tar x")

    if args.update_data == 'true':
        shutil.rmtree(MODELS_DIR)
    if not os.path.exists(MODELS_DIR) or args.update_data == 'true':
        print("pulling models")
        os.system(f"aws s3 cp {MODELS_S3} {DATA_DRIVE}")
        os.system(f"cd {DATA_DRIVE} && pbzip2 -dc models.tar.bz2 | tar x")

def gen_3d():
    train_process = subprocess.Popen(
        f"cd {REPO_DIR} &&"
        f"python3 pose_estimation/train.py "
        f"--cfg experiments/{args.config} "
        f"--log {LOG_DIR} "
        f"{args.additional_args}",
        shell=True
    )


if __name__ == '__main__':
    checkout_repo(args.git_rev)


