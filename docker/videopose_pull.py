import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser(description='Generate data for videopose')
parser.add_argument('--update_data', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--image', type=str)

args = parser.parse_args()

DATA_DRIVE = "/data/"
DATA_RAW = os.path.join(DATA_DRIVE, "raw")
DATA_RAW_H36M = os.path.join(DATA_RAW, "h36m")
DATA_RAW_MPI = os.path.join(DATA_RAW, "mpi")
DATA_RAW_H36M_S3 = "s3://geniehai/jackiey/videopose3d/raw/h36m.tar.gz"
DATA_RAW_MPI_S3 = "s3://geniehai/jackiey/videopose3d/raw/mpi.tar.gz"


def run_command(command):
    print(f"running: {command}")
    sys.stdout.flush()
    os.system(command)


def pull_any_data(data_dir, data_s3, force_update):
    parent_data_dir = os.path.dirname(data_dir)
    if force_update:
        run_command(f"rm -r {data_dir}")
        run_command(f"rm -r {parent_data_dir}")
    file_name = data_s3.rsplit('/', 1)[1]
    print(f"pulling data from {data_s3}({file_name}) to {parent_data_dir} for {data_dir}")
    run_command(f"mkdir -p {parent_data_dir}")
    run_command(f"rm {os.path.join(parent_data_dir, file_name)}")
    run_command(f"aws s3 cp --no-progress {data_s3} {parent_data_dir}")
    run_command(f"cd {parent_data_dir} && tar -xzvf {file_name} && rm {file_name}")
    run_command(f"df -h")


def pull_data():
    if args.data == "mpi_raw":
        pull_any_data(DATA_RAW_MPI, DATA_RAW_MPI_S3, args.update_data == 'true')
    elif args.data == "h36m_raw":
        pull_any_data(DATA_RAW_H36M, DATA_RAW_H36M_S3, args.update_data == 'true')


if __name__ == '__main__':
    pull_data()
