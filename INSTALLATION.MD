# Installation instruction for GPyTorch  on NVIDIA A100  GPUS

This section covers the compilation of PyTorch and related libraries. Our goal is to take advantage of the hardware acceleration that comes with NVIDIA A100 GPU 
installed at [LCRC SWING cluster](https://www.lcrc.anl.gov/systems/resources/swing/) 
and [ALCF ThetaGPU](https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview#theta-gpu). 

Now we will initialize the process by using the conda  environment file [A100_env.yml](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/A100_env.yml). 
We had some trouble with the Conda provided compilations of PyTorch with Magma + CUDA 11.00 support. So we had to do the following to get the set up working: 

```
conda env create --file A100_env.yml -p /your/path


conda activate /your/path

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install  gpytorch==1.4.1

pip install GPyOpt==1.2.6

```

We will assume that you have cloned the repo. The ECGMOD should be located and should be referenced as below in the launch script :

````
pip install -e ECGMOD

export PYTHONPATH=/your/path/lib/python3.7/site-packages/
````
