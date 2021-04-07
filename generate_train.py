from kfp import components
from kfp import dsl
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1VolumeMount,
    V1Volume,
    V1PersistentVolumeClaimVolumeSource,
    V1SecretVolumeSource,
    V1EmptyDirVolumeSource
)

from utils import add_env


def add_ssh_volume(op):
    op.add_volume(V1Volume(name='ssh-v',
                           secret=V1SecretVolumeSource(secret_name='ssh-secrets-epic-kitchen-kbbbtt9c94',
                                                       default_mode=0o600)))
    op.container.add_volume_mount(V1VolumeMount(name='ssh-v', mount_path='/root/.ssh'))
    return op


def get_container(train_op, train_env, train_num_gpus, drive='coco-headset-vol-1'):
    (train_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(train_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     .add_volume_mount(V1VolumeMount(name='data', mount_path='/data/'))
     .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
     )
    (add_env(add_ssh_volume(train_op), train_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * train_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')))
     .add_volume(V1Volume(name='data',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(drive)))
     # .add_volume(V1Volume(name='shm', host_path=V1HostPathVolumeSource(path='/dev/shm')))
     .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
     )


def get_container_no_data(train_op, train_env, train_num_gpus):
    (train_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(train_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
     )
    (add_env(add_ssh_volume(train_op), train_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * train_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')))
     # .add_volume(V1Volume(name='shm', host_path=V1HostPathVolumeSource(path='/dev/shm')))
     .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
     )


@dsl.pipeline(
    name='videopose_gen2',
    description='generate videopose data2'
)
def videopose_gen2(
        image,
        git_rev,
        resnet_model,
        mode,
        name,
        additional_args,
):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/videopose_gen.yaml')(
        image=image,
        git_rev=git_rev,
        resnet_model=resnet_model,
        mode=mode,
        name=name,
        additional_args=additional_args)
    get_container_no_data(train_op, train_env, train_num_gpus)


@dsl.pipeline(
    name='train_headset_overlay2',
    description='train_headset_overlay2'
)
def train_headset_overlay2(
        image,
        git_rev,
        update_data,
        config,
        name,
        additional_args,
):
    train_env = {}

    train_num_gpus = 4
    train_op = components.load_component_from_file('components/train.yaml')(
        image=image,
        git_rev=git_rev,
        update_data=update_data,
        config=config,
        name=name,
        additional_args=additional_args)
    get_container_no_data(train_op, train_env, train_num_gpus)


@dsl.pipeline(
    name='Train and eval epic kitchen LSTM',
    description='Train and evaluate a SlowFast + LSTM model'
)
def train_headset_overlay(
        image,
        git_rev,
        update_data,
        config,
        name,
        model_file,
        additional_args,
):
    train_env = {}

    train_num_gpus = 4
    train_op = components.load_component_from_file('components/train.yaml')(
        image=image,
        git_rev=git_rev,
        update_data=update_data,
        config=config,
        name=name,
        model_file=model_file,
        additional_args=additional_args)
    get_container(train_op, train_env, train_num_gpus)


@dsl.pipeline(
    name='videopose_pull',
    description='pull videopose data'
)
def videopose_pull(
        image,
        update_data,
        data,
):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/videopose_pull.yaml')(
        image=image,
        update_data=update_data,
        data=data)
    get_container(train_op, train_env, train_num_gpus)


@dsl.pipeline(
    name='videopose_gen',
    description='generate videopose data'
)
def videopose_gen(
        image,
        git_rev,
        resnet_model,
        mode,
        name,
        additional_args,
):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/videopose_gen.yaml')(
        image=image,
        git_rev=git_rev,
        resnet_model=resnet_model,
        mode=mode,
        name=name,
        additional_args=additional_args)
    get_container(train_op, train_env, train_num_gpus)
