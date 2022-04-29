'''

python script to deploy slurm jobs on della 

'''
import os


def compile_halos(sim):
    ''' script for training spectrum encoder
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        "#SBATCH -J compile.%s" % sim, 
        "#SBATCH --time=00:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/compile.%s.o" % sim,
        "#SBATCH --mem-per-cpu=64G",
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate uhod",
        "",
        "python /home/chhahn/projects/uhod/bin/centrals/process_data.py compile_halos %s" % sim, 
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None


def train_p_dv(sim):
    ''' deploy script for training NDE on p(dv|theta) 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J nde.p_dv.%s" % sim,
        "#SBATCH --nodes=1",
        "#SBATCH --time=47:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/npe.p_dv.%s" % sim, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/uhod/bin/centrals/train_nde.py train_p_dv %s" % sim, 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


# compile halos for tng 
compile_halos('tng') 

# train p(dv|theta) of central galaxies 
#train_p_dv('tng') 
