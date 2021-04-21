# Running a Jupyter Notebook on Compute Canada

In this tutorial we will learn how to get a Jupyter Notebook running on your local computer, *but* use Compute Canada to power the backend. Why do this? Well, you may have a large data set that requires significant amounts of RAM, storage, or processing power. You may also have a crappy computer...

The tutorial is based on:

- Compute Canada wiki page on Jupyter Notebook ([here](https://docs.computecanada.ca/wiki/JupyterNotebook))
- Youtube video from Sharcnet

## Prerequisites

- `Windows` - Make sure you have a terminal client, like [MobaXterm](https://mobaxterm.mobatek.net/download-home-edition.html) installed on your computer. If you are running Linux or MacOS, a terminal client is installed by default.

## Steps

1. **Login to Compute Canada** 

   Open your terminal client and login with your Compute Canada username.

   * Login: `<your-username>@graham.computecanada.ca`

   * Enter your password at the prompt.

   * You should now be logged in!

     
   > :memo: **Note:** The Compute Canada (CC) HPC is run in a UNIX (Linux) environment. You interact with it using typed commands. CC has a useful Wiki introducing Linux ([here](https://docs.computecanada.ca/wiki/Linux_introduction)). There are also many good resources online, like this [command cheatsheat (pdf) from fosswire](https://files.fosswire.com/2007/08/fwunixref.pdf).

2. **Create and Activate a Virtual Environment**

   Create a virtual environment that contains all the requisite Python applications to get a Jupyter Notebook up-and-running. The `virtualenv` tool allows you to easily install, and manage, Python packages.

   - Go to your home directory. The `cd` command takes you to your home directory.
   - Load the SciPy stack module: `module load scipy-stack/2020b`
     - The `scipy-stack` module includes commonly used scientific computing and data science libraries in a one-stop-shop, like Numpy, Pandas, SciPy. You can read more about modules and how to use them on the CC [wiki page on the topic](https://docs.computecanada.ca/wiki/Utiliser_des_modules/en).
     - Use the `module list` command to see which modules you currently have loaded in your environment.
   - Create the virtual environment in your home directory: `virtualenv ~/jupyter1` 
     - From now on, the`jupyter1` virtual environment will depend on loading the `scipy-stack/2020b` module first.
   - Activate the virtual environment you just created: `source ~/jupyter1/bin/activate`
     - "bin" folders, in Linux, contain ready to run  programs

3. **Install the Python Packages**

   Install the packages we need to open up a Jupyter notebook and do data analysis.



> :memo: **Note:** 

