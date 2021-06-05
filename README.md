# AL4ECG : Active Learning for Electronic Coarse Graining 


Documentation for the active learning (AL) workflow developed as a part of the article "Active Learning Strategies for ElectronicCoarse-Graining via Deep Kernel Learning". 
__For more details, please refer to the [paper](https://www.url_to_be_added.com).__


The workflow is built as a [PyTorch](https://pytorch.org) based GPU accelerated framework and offers the following capabilities:

* GPU accelerated Scalable Gaussian Processes and Exact Deep Kernel Learning (DKL) based on [GPyTorch library](https://gpytorch.ai)
* Bayesian Optimization for DKL based on GPyOpt library 
* PyTorch based numeric implementation  AL query strategy beyond standard GPR based uncertainty.



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
