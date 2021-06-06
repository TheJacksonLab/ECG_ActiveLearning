
1. For the base workflow for training a GPR or DKL at a fixed temperatures can be found in [run_ecg.py](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/workflow/run_ecg.py). The usage is listed below:


```
$ python run_ecg.py -h
usage: run_ecg.py [-h] -json JSONFILENAME [-Nt] [-is] [-qw] [-s] [-ml] [-k]
                  [-ni] [-rs]

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

```

The mandatory configuration file has the following format with a path and specification of the single temperature csv file.

```
{
"csv_name" : "/path/to/wB97x_rigid_gbcg_300K_cut.csv",
"train_range" : [0,8500],
"test_range" : [8700,9701],
"x_slice" : [0, 153],
"y_slice" : -3
}
```


Example usage with a DKL is listed below 

```
python -u  run_ecg.py  -json  config.json  -Nt 1500  -is  3   -qw 6900 -ml DKL   -s  emoc   -k  RBF  -ni 300   -rs  123   > ECG_trial.out
```


**2.  The DKL runs require a mandatory hyperparameter json file. This can be generated using Bayesian optimization**

The DKL BO script can be found in [run_DKL_Tuner.py](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/workflow/run_DKL_Tuner.py)

An example usage:

```
python -u run_DKL_Tuner.py -rs 0 -Nbo 50 100  > DKL_BO.out
```

This will create a hyperparameter json file with the following format:

```
{
    "lr": 0.0225,
    "n_iter": 500.0,
    "l1out": 600.0,
    "l2out": 250.0,
    "l3out": 200.0,
    "final": 21.0,
    "MAE": 0.5182790756225586
}
```


3. For the model training utilizing the dataset picked at mixted temperatures one must use the [run_mixedT_ECG.py](https://github.com/TheJacksonLab/ECG_ActiveLearning/blob/main/workflow/run_mixedT_ECG.py). The usage is listed below:

```
python -u  run_mixedT_ECG.py   -json  config.json  -Nt 1500  -is  3   -qw 32495  -ml DKL   -s  emoc   -k  RBF  -ni 300   -rs 123   > ECG_trial.out

```




The format for the configuration file (--json) to be used with the mixed temperature run is listed below. The hyperparameter json is frozen for a given CG resolution.



```
{
"path" : "/please/change/to/your/path",
"train_file" : "train_rigid.npy",
"test_file" : "test_rigid.npy",
"x_slice" : [0, 153], # feature dimension 
"y_slice" : -4 , # We are only using HOMO
"jump"    : {"1000K":[0, 1000],
       "650K":[1000, 2000],
       "300K":[2000, 3000],
       "50K":[3000, 4000]}
}
```
