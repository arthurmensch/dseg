import os
from os.path import expanduser

from sklearn.model_selection import ParameterGrid

job_directory = expanduser('~/output/games_rl/jobs')
if not os.path.exists(job_directory):
    os.makedirs(job_directory)
script_file = expanduser('~/work/colab/joan/games_rl/code/scripts/gan_train.py')


def write_header(i, f):
    output_file = os.path.join(job_directory, f"job_{i}.out")
    err_file = os.path.join(job_directory, f"job_{i}.err")
    f.writelines(f"#!/usr/bin/env bash\n")
    f.writelines(f"#SBATCH --job-name=train_gan\n")
    f.writelines(f"#SBATCH --output={output_file}\n")
    f.writelines(f"#SBATCH --error={err_file}\n")
    f.writelines("#SBATCH --time=20:00:00\n")
    f.writelines("#SBATCH --gres gpu:1\n")
    f.writelines("""#SBATCH --constraint "pascal&gpu_12gb"\n""")
    f.writelines("#SBATCH --qos=batch\n")
    f.writelines("#SBATCH --nodes=1\n")
    f.writelines("#SBATCH --mem=8000\n")


parameters = ParameterGrid({'lr': [1e-3, 8e-4, 5e-4],
                            'G_lr_factor': [.1], 'variance_reduction': [True, False]})
i = 0
for p in parameters:
    job_file = os.path.join(job_directory, f"job_{i}.job")
    with open(job_file, 'w+') as f:
        write_header(i, f)
        f.writelines(f"python {script_file} with extragradient_alternated lr={p['lr']}"
                     f" G_lr_factor={p['G_lr_factor']}"
                     f" variance_reduction={p['variance_reduction']}\n")

    os.system("sbatch %s" % job_file)
    i += 1

# Baseline
job_file = os.path.join(job_directory, f"job_{i}.job")
with open(job_file, 'w+') as f:
    write_header(i, f)
    f.writelines(f"python {script_file} with extragradient\n")

os.system("sbatch %s" % job_file)
