# Running a Jupyter Notebook on Compute Canada

In this tutorial we will learn how to get a Jupyter Notebook running on your local computer, *but* use Compute Canada to power the back-end. Why do this? Well, you may have a large data set that requires significant amounts of RAM, storage, or processing power. You may also have a crappy computer...

The tutorial is based on:

- Compute Canada wiki page on Jupyter Notebook ([here](https://docs.computecanada.ca/wiki/JupyterNotebook))
- Youtube video from Sharcnet ([here](https://youtu.be/5yCUDqAbBUk))

**Why use a Jupyter Notebook?** It is a useful tool that is widely used in research, data-science, and software development. They are great for prototyping and trying out new ideas.

## Prerequisites

- `Windows` - Make sure you have a terminal client, like [MobaXterm](https://mobaxterm.mobatek.net/download-home-edition.html) installed on your computer. If you are running Linux or MacOS, a terminal client is installed by default.

## Steps

1. **Login to Compute Canada** 

   Open your terminal client and login with your Compute Canada user name.

   * Login: `ssh -Y <your-username>@graham.computecanada.ca`

   * Enter your password at the prompt.

   * You should now be logged in! You are in the login node.

   > :memo: **Note:** The Compute Canada (CC) HPC is run in a Linux environment. You interact with it using typed commands. CC has a useful Wiki introducing Linux ([here](https://docs.computecanada.ca/wiki/Linux_introduction)). There are also many good resources online, like this [command cheatsheat (pdf) from fosswire](https://files.fosswire.com/2007/08/fwunixref.pdf). SciNet has a course on "The Linux Shell" (see their [training page](https://support.scinet.utoronto.ca/education/browse.php)).

   > :warning: **Warning:** Don't use the login node for computationally intensive tasks (you may get banned). The login node is for compilation and other tasks not expected to consume more than about 10 CPU-minutes and about 4 gigabytes of RAM. Otherwise, submit a job via the scheduler, or request an allocation.

2. **Download Tutorial Files from GitHub**

   We need the tutorial files. They contain  data we want to explore!

   * Navigate to your projects folder. Something like: `cd projects/def-mechefsk/<your_username>`
   * Now clone the tutorial [Github repository](https://github.com/tvhahn/compute-canada-hpc). Type: `git clone https://github.com/tvhahn/compute-canada-hpc.git`
     * Git is already installed on the Compute Canada system (it comes by default in most linux distributions)

   > :bulb: **Tip:** Git is an important tool in modern software development. Start using it today! Get yourself a [github](https://github.com/) account. [Download git](https://git-scm.com/download/win) (if you're on Windows). Here's a simple git guide that I use almost every day: [git - the simple guide](http://rogerdudler.github.io/git-guide/)

3.  **Create and activate a virtual environment**

   Create a virtual environment that contains all the requisite Python modules to get a Jupyter Notebook up-and-running. The `virtualenv` tool allows you to easily install, and manage, Python packages.

   We will be using TensorFlow in the demo. There are some compatibility issues between TensorFlow and some other packages when using the standard environment on Compute Canada. To address this, we will load Python 3.6.  🤷

   - Go to your home directory. The `cd` command takes you to your home directory.
   - Load Python 3.6:  `module load python/3.6` 
     - Before you create a virtual environment, make sure you have the proper version of Python selected.
     - Use the `module list` command to see which modules you currently have loaded in your environment. Use `module unload <module_name>` to unload a module.
   - Create the virtual environment in your home directory: `virtualenv ~/jupyter1` 
   - Activate the virtual environment you just created: `source ~/jupyter1/bin/activate`
     - "bin" folders, in Linux, contain ready to run  programs
     - To deactivate a virtual environment, use the command `deactivate`

   > :bulb: **Tip:** If you are just doing data exploration, you can use the `scipy-stack/2020b` module. It includes commonly used scientific computing and data science libraries in a one-stop-shop, like Numpy, Pandas, and SciPy. You can read more about modules and how to use them on the CC [wiki page on the topic](https://docs.computecanada.ca/wiki/Utiliser_des_modules/en).

4. **Install the Python packages**

   Install the packages we need to open up a Jupyter notebook and do data analysis.

   * While the `jupyter1` environment is active, upgrade the package manager, pip: `pip install --no-index --upgrade pip` You should always do this when setting up a new environment.
     
   * Install basic data-science packages, scikit-learn, Pandas, Matplotlib: `pip install --no-index pandas scikit_learn matplotlib seaborn`
     
        > :bulb: **Tip:** Compute Canada has many common python packages already compiled (made into "wheels") on their system (see available [python wheels](https://docs.computecanada.ca/wiki/Available_Python_wheels)). These are installed with pip using the `--no-index` command. Installing the wheels from CC can save considerable time, and prevent compatibility issues.
     
   * Install TensorFlow 2.0 and Jupyter Lab: `pip install --no-index tensorflow jupyterlab`

5. **Create a script to launch Jupyter Lab** 

   Use nano (text editor in linux) to create a bash script that we'll call upon to open up a Jupyter Lab session.

   * Create a script in your virtual environment (make sure `jupyter1` is active), in the bin folder: `nano $VIRTUAL_ENV/bin/notebook.sh`

   * This opens up the nano text editor, so that we can create the bash script (see the [Youtube video](https://youtu.be/5yCUDqAbBUk?t=969) for more details):

       ```bash
       #!/bin/bash
       unset XDG_RUNTIME_DIR
       jupyter-lab --ip $(hostname -f) --no-browser
       ```

       Press ctrl-O to save, ctrl-X to exit. 

   * Back in your home directory, change the user privileges of the `notebook.sh` that you just created (we'll allow the user, *u*, to execute, *x*, the file). This is needed so that we can run the script in the bin folder: `chmod u+x $VIRTUAL_ENV/bin/notebook.sh `
   
6. **Create an allocation to run Jupyter Lab**

   While in your virtual environment, run the following:

   * ```
     salloc --time=1:0:0 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=2048M --account=def-profaccount srun $VIRTUAL_ENV/bin/notebook.sh
     ```
     
     * Allocate 1 hour for 1 task, using 4 cpus and 2048 MB of RAM/CPU. Allocated on the
     
       > :warning: **Warning:** Try not to allocate more than you need so that the resources can be efficiently used between users.
     
   * When you have the allocation, you should see something like this:

     ![terminal_notebook](./images/terminal_notebook.png)

7. **SSH tunnel from your local computer into the Jupyter Notebook**

   The Jupyter Notebook is now running on the Compute Canada HPC. We need to "tunnel" into the HPC system and show the notebook on our local computer.

   * Open a new terminal window.

   * In the new terminal, ssh into the graham server. Type something like this, based on what is shown the other terminal you have open showing the notebook access token:

     ``` 
     ssh -L 8888:gra105.graham.sharcnet:8888 tvhahn@graham.computecanada.ca
     ```
     * The local port is 8888. The local host will be port forwarding the 8888 port to the gra105.sharcnet:8888 port.

   * It will ask you for your login credentials. Fill that in.

   * Then on local browser, copy the link to Jupyter lab with the access token, like: `http://localhost:8888/?token=<token>`. Or, you can copy the link from your terminal (or click it if your terminal client allows you to).

8. **Run Notebooks**

   Now you can make and run notebooks! Notebooks are a great way to explore data, and prototype code. As an example, if you are data science work, this would be a good workflow (from [pytorch-style guide](https://github.com/IgorSusmelj/pytorch-styleguide)):

   > 1. Start with a Jupyter notebook
   > 2. Explore the data. Prototype models.
   > 3. Build your classes/ methods inside cells of the notebook
   > 4. Move your code to python scripts
   > 5. Train / deploy on server (Compute Canada in our case)

| **Jupyter Notebook** | **Python Scripts** |
|----------------------|--------------------|
| + Exploration | + Running longer jobs without interruption |
| + Debugging | + Easy to track changes with git |
| - Can become a huge file| - Debugging mostly means rerunning the whole script|
| - Can be interrupted (don't use for long training) | |
| - Prone to errors and become a mess | |

If you've already cloned the repo, you can navigate into the `compute-canada-hpc` folder and open up the notebooks within it.

> :memo: **Note**: You can also run your IDE (interactive developer environment), such as VS Code, on the Compute Canada system. Same with Matlab!

