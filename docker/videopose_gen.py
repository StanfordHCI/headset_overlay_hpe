import argparse
import os
import shutil
import subprocess
import sys
import time

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
RESNET_MODEL_PATH = "/workspace/resnetpose/weights/model_best.pth.tar"
RESNET_REPO_PATH = os.path.join(REPO_DIR, "resnet.pth")

h36m_3d_keypoints = "s3://geniehai/jackiey/videopose3d/data_3d_h36m_new2.npz"

RESNET_REPO_DIR = "/workspace/resnetpose"
RESNET_WEIGHT_DIR = os.path.join(RESNET_REPO_DIR, "weights")

DATA_DRIVE = "/data/"
DATA_RAW = os.path.join(DATA_DRIVE, "raw")
DATA_RAW_H36M = os.path.join(DATA_RAW, "h36m")
DATA_RAW_MPI = os.path.join(DATA_RAW, "mpi")


def run_command(command):
    print(f"running: {command}")
    sys.stdout.flush()
    os.system(command)


def get_resnet_model():
    run_command(f"mkdir -p /workspace/resnetpose/weights/")
    run_command(f"rm {RESNET_MODEL_PATH}")
    run_command(f"aws s3 cp --no-progress {RESNET_MODEL_S3} {RESNET_MODEL_PATH}")


def checkout_repo(git_rev: str):
    run_command(f"cd {REPO_DIR} && git pull && git checkout {git_rev}")


def gen_3d_h36m():
    gen_process = subprocess.Popen(
        f"cd {REPO_DIR}/data &&"
        f"python3 prepare_data_h36m.py "
        f"--from-source-cdf {DATA_RAW_H36M} "
        f"{args.additional_args}",
        shell=True
    )
    gen_process.wait()


def gen_2d_h36m():
    command = \
        f"cd {REPO_DIR}/data &&" \
        f"python ./prepare_data_2d_h36m_resnet.py " \
        f"--from-source-cdf {DATA_RAW_H36M} " \
        f"{args.additional_args}"
    print(f"running: {command}")
    gen_process = subprocess.Popen(
        command,
        shell=True
    )
    gen_process.wait()


def sync_s3():
    run_command(
        f'cd {REPO_DIR}/data && aws s3 cp --no-progress ./ {S3_OUTPUT_DIR} --recursive --exclude "*" --include "data_*"')


def get_3d_points():
    if args.mode == "2d_h36m":
        run_command(f"cd {REPO_DIR}/data && aws s3 cp --no-progress {h36m_3d_keypoints} ./")


if __name__ == '__main__':
    checkout_repo(args.git_rev)
    checkout_repo(args.git_rev)
    if args.mode == "3d_h36m":
        gen_3d_h36m()
    elif args.mode == "2d_h36m":
        get_3d_points()
        get_resnet_model()
        gen_2d_h36m()

    sync_s3()
    # time.sleep(60 * 60)
