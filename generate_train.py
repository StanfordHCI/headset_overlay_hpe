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


@dsl.pipeline(
    name='Train and eval epic kitchen LSTM',
    description='Train and evaluate a SlowFast + LSTM model'
)
def train_eval_epic(
        owner,
        project,
        experiment,
        model,
        git_rev,
        pretrained_s3,
        mode,
        train_additional_args='',
        eval_additional_args=''
):
    train_env = {}

    train_num_gpus = 4
    train_op = components.load_component_from_file('components/train.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        git_rev=git_rev,
        pretrained_s3=pretrained_s3,
        mode=mode,
        additional_args=train_additional_args)
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
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('coco-headset-vol-1')))
     # .add_volume(V1Volume(name='shm', host_path=V1HostPathVolumeSource(path='/dev/shm')))
     .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
     )

    # eval_env = {}

    # eval_op = components.load_component_from_file('components/evaluate-sumbt.yaml')(
    #         owner=owner,
    #         project=project,
    #         experiment=experiment,
    #         s3_datadir=s3_datadir,
    #         s3_model_dir=train_op.outputs['s3_model_dir'],
    #         additional_args=eval_additional_args)
    # (eval_op.container
    #     .set_memory_limit('15Gi')
    #     .set_memory_request('15Gi')
    #     .set_cpu_limit('4')
    #     .set_cpu_request('4'))
    # add_env(add_ssh_volume(eval_op), eval_env)