"""
@author: gsivaraman@anl.gov
"""
from __future__ import print_function
from pprint import pprint
from ECGMOD.preprocess import prepare_data, return_tensor
from ECGMOD.OTFSample import OTFSampler
from ECGMOD.GPmodels import InferGPModel
from ECGMOD.DKLTuner import DKLmodel
from ECGMOD.ALquery import *
from ECGMOD.postprocess import plot_metric_write
import argparse
import random
import GPyOpt
import json
import numpy as np
import torch 
import os 


def get_argparse():
    """
    A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(
        description='A python workflow to run BO tuner  for Electronic Coarse Graining DKL model')
    parser.add_argument('-rs', '--randomseed', type=int, metavar='',
            help="Initialize the random seed", default=0)
    parser.add_argument('-Nbo', '--Nopt', nargs='+', type=float, required=True,
                        help="Number of exploration and optimization steps for BO. e.g.'-Nbo 25 50' ")
    return parser.parse_args()


def f(x):
    """
    Surrogate function over the error metric to be optimized
    """
   
    if os.path.isfile("settings.json"):
        settings_json = json.load(open('settings.json'))
    else:
        raise Exception("**ERROR** BO settings json  file not found! File must be provided!")

    jsonfile = settings_json["json"]
    Nseed = settings_json["Nseed"]
    method = settings_json["GP"]
    kernel = settings_json["kernel"]
    rand_seed = settings_json["RandSeed"]
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    train_x, train_y, test_x, test_y, numy, scaleY, scalerX = prepare_data(
        jsonfile, NormalizeY=True, NormalizeX=True)
    train_ind = [i for i in range(len(train_x))]
    test_ind = [j for j in range(len(test_x))]

    indlist = [] 
    sampler = OTFSampler(index=train_ind)
    reduced_original, init = sampler._randsampler(width=Nseed)
    [indlist.append(val) for val in init if val not in indlist]

    print("\n..............................\n")

    x_train, x_test, y_train, y_test = return_tensor(
        train_x, test_x, train_y, test_y, indlist)

    mse, r2, _, _, _, _ = DKLmodel(x_train, x_test, y_train, y_test, numy=1, n_iter=int(x[:, 1]),
     verbose=True, GP=method, kernel=kernel, lr=float(x[:, 0]),
     l1out=int(x[:, 2]), l2out=int(x[:, 3]), l3out=int(x[:, 4]), final=int(x[:, 5]))

    print("\nParam: {}, {}, {}, {}, {}, {}  |  MAE : {}, R2: {}".format(float(x[:, 0]), float(x[:, 1]), float(x[:, 2]), float(x[:, 3]),
    float(x[:, 4]), float(x[:, 5]), mse, r2))

    return mse


def main():
    '''
    Gather all the ECG-AL workflow here!
    '''

    args = get_argparse()
    Nopt = tuple(args.Nopt)
    random.seed(args.randomseed)  # Initialize the random seed

    # print("\n..............................\n")

    bounds = [{'name': 'lr',          'type': 'discrete',  'domain': np.linspace(0.0001,0.1,1000) },
    {'name': 'n_iter',            'type': 'discrete', 'domain': range(50, 1001, 50)},
    {'name': 'l1out',        'type': 'discrete', 'domain': np.arange(500, 1001, 50)},
    {'name': 'l2out',            'type': 'discrete','domain': np.arange(200, 501, 50)},
    {'name': 'l3out',            'type': 'discrete', 'domain': np.arange(50, 201,50)},
    {'name': 'final',            'type': 'discrete',  'domain': np.arange(2, 31)}, ]

    opt_quip = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=int(Nopt[0]),
                                             model_type="GP_MCMC",
                                             acquisition_type='EI_MCMC',  # EI
                                             evaluator_type="predictive",  # Expected Improvement
                                             exact_feval=False,
                                             maximize=False)  # --> True only for r2

    print("\nBegin Optimization run \t")
    opt_quip.run_optimization( max_iter= int(Nopt[1]) )

    hyperdict = {}

    for num in range(len(bounds)):
        hyperdict[bounds[num]["name"]]=opt_quip.x_opt[num]
    hyperdict["MAE"]=opt_quip.fx_opt
    print("\nBest hyperparameters : ")
    pprint(hyperdict)
    with open('hyperparameters.json', 'w', encoding='utf-8') as outfile:
        json.dump(hyperdict, outfile, ensure_ascii=False, indent=4)
    print("\n..............................\n")


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
