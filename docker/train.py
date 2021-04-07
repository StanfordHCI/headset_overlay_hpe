import argparse
import os
import shutil
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--git_rev', type=str)
parser.add_argument('--update_data', type=str)
parser.add_argument('--config', type=str)  # coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
parser.add_argument('--model_file', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--additional_args', type=str)

args = parser.parse_args()

REPO_DIR = "/workspace/human-pose-estimation.pytorch"
OUTPUT_DIR = os.path.join(REPO_DIR, 'output')
PROJECT = "coco_headset"
OWNER = "jackiey"
EXPERIMENT = args.name
MODEL = args.config.rpartition('.')[0]
DATETIME = datetime.now().strftime('%Y%m%dT%H%M%S')
LOG_DIR = f"/shared/tensorboard/{PROJECT}/{EXPERIMENT}/{OWNER}/{MODEL}"
S3_MODEL_DIR = f"s3://geniehai/{OWNER}/models/{PROJECT}/{EXPERIMENT}/{MODEL}/{DATETIME}/"

DATA_DRIVE = "/data/"
DATA_S3 = "s3://geniehai/jackiey/coco_headset/data.tar.bz2"
MODELS_S3 = "s3://geniehai/jackiey/coco_headset/models.tar.bz2"
DATA_DIR = os.path.join(DATA_DRIVE, "data")
MODELS_DIR = os.path.join(DATA_DRIVE, "models")
REPO_DATA_DIR = os.path.join(REPO_DIR, "data")
REPO_MODELS_DIR = os.path.join(REPO_DIR, "models")
MODEL_PATH = os.path.join(DATA_DRIVE, "hpe.model")


def checkout_repo(git_rev: str):
    os.system(f"cd {REPO_DIR} && git pull && git checkout {git_rev}")


def pull_data():
    os.system("mkdir -p /data")
    if args.update_data == 'true':
        shutil.rmtree(DATA_DIR)
    if not os.path.exists(DATA_DIR) or args.update_data == 'true':
        print("pulling data")
        os.system(f"aws s3 cp {DATA_S3} {DATA_DRIVE}")
        os.system(f"cd {DATA_DRIVE} && pbzip2 -dc data.tar.bz2 | tar x && rm data.tar.bz2")

    if args.update_data == 'true':
        shutil.rmtree(MODELS_DIR)
    if not os.path.exists(MODELS_DIR) or args.update_data == 'true':
        print("pulling models")
        os.system(f"aws s3 cp {MODELS_S3} {DATA_DRIVE}")
        os.system(f"cd {DATA_DRIVE} && pbzip2 -dc models.tar.bz2 | tar x && rm models.tar.bz2")


def acquire_model():
    os.system(f"aws s3 cp {args.model_file} {MODEL_PATH}")


def setup_dir():
    os.mkdir(OUTPUT_DIR)
    os.symlink(DATA_DIR, REPO_DATA_DIR)
    os.symlink(MODELS_DIR, REPO_MODELS_DIR)


def sync_s3():
    os.system(f"aws s3 sync --no-progress {OUTPUT_DIR} {S3_MODEL_DIR}")


class OutputUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print('output dir updated, sync to s3.')
        sync_s3()


if __name__ == '__main__':
    pull_data()
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    checkout_repo(args.git_rev)
    setup_dir()
    event_handler = OutputUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, path=OUTPUT_DIR, recursive=True)
    observer.start()

    if args.model_file != "":
        acquire_model()
        train_process = subprocess.Popen(
            f"cd {REPO_DIR} &&"
            f"python3 pose_estimation/valid.py "
            f"--cfg experiments/{args.config} "
            f"--model-file experiments/{args.config} "
            f"--log {LOG_DIR} "
            f"{args.additional_args}",
            shell=True
        )
    else:
        train_process = subprocess.Popen(
            f"cd {REPO_DIR} &&"
            f"python3 pose_estimation/train.py "
            f"--cfg experiments/{args.config} "
            f"--log {LOG_DIR} "
            f"{args.additional_args}",
            shell=True
        )

    train_process.wait()
    sync_s3()
