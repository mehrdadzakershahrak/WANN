# Introduction
We believe that agent research should be for everyone to benefit from. Similar to the computer revolution,
we are sitting at a time in history where we truly are heading down the road that one day will lead to 
artificial general intelligence and potentially forms of realized singularity.

To that end our goal is not only for AGI but for what it can do to better the lives of those that need it, as well
as to impact the human race so that we can focus more on the things in life that really matter.

This repo is an MIT Licensed source and always will be. Feel free to use this in any research publications with 
proper citation listed here:

as well as in any commercial product that bring benefits to mankind.

#INSTALLATION

# TODO: complete me

# Installation of PyTorch on windows platform(s)
pip install torch==1.5.1 torchvision -f https://download.pytorch.org/whl/torch_stable.html

# TEMP INSTALLATION NOTES. TODO: Clean me up
activate conda env

conda install -c conda-forge ffmpeg

pip install git+https://github.com/Kojoley/atari-py.git
 
 cd baselines
 pip install -e .
 
 
 Windows Unofficial Installation Support
 https://arztsamuel.github.io/en/blogs/2018/Gym-and-Baselines-on-Windows.html
 
 
 Install Microsoft MPI
https://www.microsoft.com/en-us/download/details.aspx?id=54607

You need to run both files msmpisdk.msi and MSMpiSetup.exe

Add $PATH$ in the Environment Variables.
$PATH$ is where you install the Microsoft MPI. In my case:

C:\Program Files (x86)\Microsoft SDKs\MPI

Install package in Anaconda
conda install mpi4py
or
pip install mpi4py

And finally, test the installation
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print (“hello world from process “, rank)


python 3.7 or lower required


## ubuntu

sudo apt-get --assume-yes install python3 python3-pip git zlib1g-dev libopenmpi-dev ffmpeg
sudo apt-get update

pip3 install --timeout 1000 opencv-python anyrl gym-retro gym joblib atari-py tensorflow

git clone https://github.com/openai/baselines.git
cd baselines
pip3 install -e .

