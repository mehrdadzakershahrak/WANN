# WEIGTH AGNOSTIC NEURAL NETWORK FAST CONVERGENCE PROJECT

## INSTALLATION

1) execute 'pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html'

2)
2.1) git clone https://github.com/benelot/pybullet-gym.git
2.2) cd pybullet-gym
2.3) pip install -e .

3) cd to the base directory and execute pip install -r requirements.txt

## INSTALL MPI

# ON MAC
brew install open-mpi

# ON WINDOWS
Download latest MPI from Microsofts site (e.g. https://www.microsoft.com/en-us/download/details.aspx?id=57467)
Append the bin location to the PATH variable

# ON LINUX
sudo apt-get update -y
sudo apt-get install -y openmpi-bin

## EXPERIMENT SETUP

Configuration modification can be made from each corresponding task config file (i.e. task/foo)
or from the config.py file directly

## RUN

1) execute 'python main.py'

## CONTRIBUTION
TBD