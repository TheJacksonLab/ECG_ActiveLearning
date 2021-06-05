# AL4ECG : Active Learning for Electronic Coarse Graining 


Documentation for the active learning (AL) workflow developed as a part of the article "Active Learning Strategies for ElectronicCoarse-Graining via Deep Kernel Learning". 
__For more details, please refer to the [paper](https://www.url_to_be_added.com).__



## What are the capabilities of AL4ECG workflow ?

The workflow is built as a [PyTorch](https://pytorch.org) based GPU accelerated framework and offers the following capabilities:

* GPU accelerated Scalable Gaussian Processes and Exact Deep Kernel Learning (DKL) based on [GPyTorch library](https://gpytorch.ai)
* Bayesian Optimization for DKL based on GPyOpt library 
* PyTorch based numeric implementation of AL query strategy beyond standard GPR based uncertainty.
* Capable of running on the state-of-the-art [NVIDIA A100 GPU's](https://www.nvidia.com/en-us/data-center/a100/).

## Installation 

Running GPyTorch on A100 GPU has the following basic requirments:

* CUDA 11.0
* MAGMA support for CUDA 11.0

The step-by-step compilation is covered in [INSTALLATION.MD](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/INSTALLATION.MD)


## How do i cite AL4ECG workflow ?

If you are using this active learning workflow  in your research paper, please cite us as
```
@article{AL4ECG,
  title={Active Learning Strategies for ElectronicCoarse-Graining via Deep Kernel Learning},
  author={Sivaraman, Ganesh and Jackson, Nicholas},
  journal={XX},
  volume={YY},
  number={ZZ},
  pages={BB},
  year={2021},
  publisher={ Publishing Group}
}
```

## Acknowledgements
This  material  is  based  upon  work  supported  by  Laboratory  Directed  Research  and  Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.
