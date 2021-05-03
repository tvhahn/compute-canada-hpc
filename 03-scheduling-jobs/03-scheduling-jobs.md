# Scheduling Jobs on Compute Canada

In this tutorial we will learn how to get a Jupyter Notebook 

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

2. 

