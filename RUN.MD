



For the model training utilizing the dataset picked at mixted temperatures one must use the [run_mixedT_ECG.py](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/workflow/run_mixedT_ECG.py). The usage is listed below:


````
$ python run_mixedT_ECG.py -h
usage: run_mixedT_ECG.py [-h] -json JSONFILENAME [-Nt] [-is] [-qw] [-s] [-ml]
                         [-k] [-ni] [-rs]

A python workflow to run active learning for Electronic Coarse Graining

optional arguments:
  -h, --help            show this help message and exit
  -json JSONFILENAME, --jsonfilename JSONFILENAME
                        Input JSON configuration filename (including path)
  -Nt , --ntrials       Number of AL trials to run
  -is , --initialseed   Number of initial random points to bootstrap the model
  -qw , --querywidth    AL query width
  -s , --strategy       AL strategy : 'random, uncertainty, emoc'
  -ml , --mlmethod      GP type: EXACT, DKL, MULTITASK
  -k , --kernel         GP Kernel: RBF, MATERN
  -ni , --niter         Number of iteration for converging GP
  -rs , --randomseed    Initialize the random seed
````
