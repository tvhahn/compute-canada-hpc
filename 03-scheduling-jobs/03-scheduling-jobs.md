# Scheduling Jobs on Compute Canada

In this tutorial we will schedule a job using the scheduling system.

Based on tutorial form WestGrid ([here](https://www.youtube.com/watch?v=RCodAqGlFeM)) and ACENET ([here](https://youtu.be/ahhZb8Onk-k)).

We'll run through some slides from the tutorial [Intro into Running Jobs](https://westgrid.github.io/trainingMaterials/materials/introRunningJobs20180919.pdf) first.

**Why schedule jobs?** As noted in the previous section, notebooks are not suitable for running larger jobs. A user can schedule jobs to run on the HPC system. This is done by defining the parameters of a job (how much compute you need) and the time for the job to run.



> :warning: **Warning:** Do not schedule more resources that what you require to finish the job. 

> :bulb: **Tip:** Smaller jobs (e.g. 1 hour) will be quicker to run than large jobs that require many resources and a long time to run. Also, if you have many smaller jobs, you can try to run these on the Rapid-Access-Service, as opposed to the allocation award.



## Prerequisites

- `Windows` - Make sure you have a terminal client, like [MobaXterm](https://mobaxterm.mobatek.net/download-home-edition.html) installed on your computer. If you are running Linux or MacOS, a terminal client is installed by default.
- Make sure you've cloned this repository to your project folder. 
  - `git clone https://github.com/tvhahn/compute-canada-hpc.git`

## Steps

1. **Login to Compute Canada** 

   Open your terminal client and login with your Compute Canada user name.

   * Login: `ssh -Y <your-username>@graham.computecanada.ca`
* Enter your password at the prompt.
   * You should now be logged in! You are in the login node.

2. **Navigate to the Schedule Job Folder**

   Make your way to the `03-scheduling-jobs` folder

3. **Run the bash script**

   We have two bash scripts in the folder, one for training models using a GPU, and one for using CPUs.

   The `random_search_cpu.sh` is as follows:

   ```bash
   #!/bin/bash
   #SBATCH --account=rrg-mechefsk
   #SBATCH --cpus-per-task=4   # number of cores
   #SBATCH --mem=4G            # memory for the entire job across all cores (4GB)
   #SBATCH --time=0-00:10      # time (DD-HH:MM)
   #SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
   #SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
   #SBATCH --mail-user=your_username@queensu.ca   # Email to which notifications will be $
   
   module load python/3.6
   source ~/jupyter1/bin/activate
   
   python train_model_tcn.py
   ```

   The  `random_search_gpu.sh` is as follows:

   ```bash
   #!/bin/bash
   #SBATCH --account=rrg-mechefsk
   #SBATCH --gres=gpu:1        # request GPU "generic resource"
   #SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
   #SBATCH --mem=12G           # memory per node
   #SBATCH --time=0-00:10      # time (DD-HH:MM)
   #SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
   #SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
   #SBATCH --mail-user=your_username@queensu.ca   # Email to which notifications will be $
   
   module load python/3.6
   source ~/jupyter1/bin/activate
   
   python train_model_tcn.py
   ```

   * We will run the `random_search_cpu.sh` instead of the GPU job (want to conserve GPU resources). In the terminal, type: `sbatch random_search_cpu`
   * You can find the jobs you have submitted by typing `squeue -u <your_username>`
   * You can cancel all your submitted jobs by typing `scancel -u <your_username>`

   > :bulb: **Tip:** You should always "test" out your jobs before running them. There will be bugs. Start with a very small sub-set of the data, and a short run-time, and see if your programs work. You can also request a small allocation with a few CPUs, `salloc`, to test out your scripts.



## Running Array Jobs

Array jobs are similar to single batch jobs, except that they call the same Python script (or whatever script) multiple time. They allocate each run of the script to a different core. If you are going to be running the same script multiple times, but there are very few changes between each run, use an array job.

**Example:** I've used array jobs to generate multiple features (like RMS, kurtosis FFT peaks, etc.) on ~300,000 unique signals from a CNC machine. To create all the features on my local machine it **took 3 days**. I sped it up by using array jobs and was able to create all the features in **1 hour**.

How?

* Took all the unique cut signals, and divided them up into ~100 zip files (see the `zip_files.py`) The names of each zip file was saved into a file called `input_zip_files` which would look like:

  ```
  0.zip
  1.zip
  2.zip
  3.zip
  4.zip
  5.zip
  ...
  ```

  

* Each array job called on one of the zip files (via the `input_zip_files`), extracted the contents of the zip to `$scratch`, and calculated the features. The calculated features were saved to a CSV in the scratch folder. These individual csv's were combined later.

  Here's what the bash script for calling the array job looked like:

  ```bash
  #!/bin/bash
  #SBATCH --time=01:00:00 	# 1 hour
  #SBATCH --array=1-111 		# 111 jobs. job 1 would call on line one in input_zip_files
  #SBATCH --mem=2G 			# Each job only needs 1 cpu and 2GB of ram
  #SBATCH --mail-type=ALL     # Type of email notification- BEGIN,END,F$
  #SBATCH --mail-user=your_email@queensu.ca   # Send email notifications
  ## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays
  
  echo "Starting task $SLURM_ARRAY_TASK_ID"
  DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input_zip_files)
  
  module load scipy-stack/2019b
  python create_features.py $DIR
  ```

  

## Best Practices

Look at the best practices section from the [Intro into Running Jobs](https://westgrid.github.io/trainingMaterials/materials/introRunningJobs20180919.pdf) pdf.



:bulb: **Tip Regarding Python Jobs in Arrays**, from https://docs.computecanada.ca/wiki/Python

>Parallel filesystems such as the ones used on our clusters are very  good at reading or writing large chunks of data, but can be bad for  intensive use of small files. Launching a software and loading  libraries, such as starting python and loading a virtual environment,  can be slow for this reason. 
>
>As a workaround for this kind of slowdown, and especially for  single-node Python jobs, you can create your virtual environment inside  of your job, using the compute node's local disk. It may seem  counter-intuitive to recreate your environment for every job, but it can be faster than running from the parallel filesystem, and will give you  some protection against some filesystem performance issues. This  approach, of creating a node-local virtualenv, has to be done for each  node in the job, since the virtualenv is only accessible on one node.   Following job submission script demonstrates how to do this for a  single-node job: 

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python ...
```

