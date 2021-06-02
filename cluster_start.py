import os,sys,shutil
import subprocess
import moxing as mox

# config working dir
WORK_DIR='/home/work/user-job-dir/uvpsandbox/'  #uspsandbox is the program name.
os.chdir(WORK_DIR)
print("currrent working dir: ", os.getcwd())

# package install
print('update torch ....................')
print('install requirements ....................')
shell_command = 'pip install -r train_scripts/requirements.txt'
out_bytes = subprocess.check_output(shell_command, shell=True)
# install cupy
shell_command = 'CUDA_PATH=/usr/local/cuda; export CUDA_PATH;  pip install train_scripts/shell/cupy_cuda100-8.5.0-cp36-cp36m-manylinux1_x86_64.whl'
out_bytes = subprocess.check_output(shell_command, shell=True)

out_text = out_bytes.decode('utf-8')
print(out_text)

# code init
try:
    out_bytes = subprocess.check_output('bash init.sh', shell=True)
except subprocess.CalledProcessError as e:
    out_bytes = e.output       # Output generated before error
    code      = e.returncode   # Return code

out_text = out_bytes.decode('utf-8')

print(out_text)

# data copy
#CODEROOT='s3://modelarts-ad/q00293849/projects/MultiObjectiveOptimization/'
#REMOTE_DATAROOT = 's3://modelarts-ad/q00293849/data/datasets/cityscapes_mini/'
bucket='bucket-adas-shanghai'
REMOTE_DATAROOT = 's3://{}/z00519457/Data/Cityscapes/'.format(bucket)
LOCAL_DATAROOT = '/cache/cityscapes/' #All the files can only be stored in /cache
DATA_FOLDERS = ['leftImg8bit/', 'panoptic/', 'gtBbox3d']

# PRETRAINED_MODEL='optimizer=SGD_lr=0.02_tasks=SID_normalization_type=none_algorithm=mgda_sequence_coding=True_save_suffix=gaussian769_mgda_66_model.pkl'
# PRETRAINED_PATH=os.path.join('s3://modelarts-ad/q00293849/data/output/mtl/mtlv1.1_gaussian769_mgda/saved_models/', PRETRAINED_MODEL)

print("remote data dir:", REMOTE_DATAROOT)
print("local data dir:", LOCAL_DATAROOT)

# if not os.path.exists('./saved_models'):
#     os.makedirs('./saved_models')
#mox.file.copy(PRETRAINED_PATH, './saved_models/checkpoint.pkl')

print("downloading data...")
if DATA_FOLDERS is None:
    mox.file.copy_parallel(REMOTE_DATAROOT, LOCAL_DATAROOT)
elif len(DATA_FOLDERS) > 0:
    for folder in DATA_FOLDERS:
        mox.file.copy_parallel(REMOTE_DATAROOT + folder, LOCAL_DATAROOT + folder)

# if not os.path.exists('./data'):
#     os.makedirs('./data')
# subprocess.call('ln -s {} ./data/cityscapes'.format(LOCAL_DATAROOT), shell=True)

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
# training
command_list = ['python uvpsandbox/multitask/train_test.py',
           '--output_path=/cache/output/',
           '--output_prefix=mnv3_pd_centernet_smoke_interact_a15_cosine_from_nointermtl',
           '--remote_output_path=s3://{}/q00293849/output/uvpsandbox/'.format(bucket),
           '--serialize.load_path=model/Iter-58000_finetunefrom_a3_anchor33_35_33.pth',
           '--solver.grad_norm_alpha=1.5',
           '--solver.task_loss_weight=[1.,1.,1.]',
           '--solver.loss_anchor=[0.33,3.5,0.33]',
           '--solver.loss_window=100',
           '--solver.lr_schedule=cosine_decay',
           '--solver.lr=0.001',
           '--solver.warmup_iter=1000',
           '--solver.decay_its=[10000，100000,]',
           '--solver.max_it=50000',
           '--common_cfg=uvpsandbox/experiments_mtl/common_pd_centernet_smoke@cityscapes.yaml',
           '--task_cfg=uvpsandbox/experiments_mtl/pd_centernet_smoke@cityscapes/cityscapes_mnv3_pd_interact.yaml',
           '--task_cfg=uvpsandbox/experiments_mtl/pd_centernet_smoke@cityscapes/cityscapes_mnv3_2d_interact.yaml',
           '--task_cfg=uvpsandbox/experiments_mtl/pd_centernet_smoke@cityscapes/cityscapes_mnv3_3d.yaml',
           ]

command = ' '.join(command_list)
subprocess.call(command, shell=True)
