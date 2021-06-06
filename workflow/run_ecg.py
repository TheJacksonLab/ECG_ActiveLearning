from ECGMOD.preprocess import prepare_data, return_tensor
from ECGMOD.OTFSample import OTFSampler
from ECGMOD.GPmodels import GPmodel, InferGPModel 
from ECGMOD.DKLTuner import DKLmodel 
from ECGMOD.ALquery import *
from ECGMOD.postprocess import plot_metric_write
import argparse 
from tqdm import tqdm
import random , json, os
import torch
import numpy as np 

def get_argparse():
    """
    A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(description='A python workflow to run  active learning for Electronic Coarse Graining')
    parser.add_argument('-json','--jsonfilename', type=str,required=True,help='Input JSON configuration filename (including path)')
    parser.add_argument('-Nt','--ntrials',type=int,metavar='',\
                       help="Number of AL trials to run",default=100)
    parser.add_argument('-is','--initialseed',type=int,metavar='',\
                               help="Number of initial random points to bootstrap the model",default=10)
    parser.add_argument('-qw','--querywidth',type=int,metavar='',\
                               help="AL query width",default=500)
    parser.add_argument('-s','--strategy',type=str,metavar='',\
            help="AL strategy : 'random, uncertainty, emoc' ",default='uncertainty')
    parser.add_argument('-ml','--mlmethod',type=str,metavar='',\
            help="GP type: EXACT, DKL, MULTITASK",default='EXACT')
    parser.add_argument('-k','--kernel',type=str,metavar='',\
            help="GP Kernel: RBF, MATERN",default='MATERN')
    parser.add_argument('-ni','--niter',type=int,metavar='',\
            help="Number of iteration for converging GP",default=500)
    parser.add_argument('-rs','--randomseed',type=int,metavar='',\
            help="Initialize the random seed",default=0)
    return parser.parse_args()



def main():
    '''
    Gather all the ECG-AL workflow here! 
    '''

    args = get_argparse()     
    csv = args.jsonfilename
    Ntrials = args.ntrials
    Nseed = args.initialseed
    
    random.seed(args.randomseed)  # Initialize the random seed 
    np.random.seed(args.randomseed) 
    print("\nSet random seed :", args.randomseed )    
 
    train_x, train_y, test_x, test_y, numy, scaleY, scalerX = prepare_data(csv,NormalizeY=True,NormalizeX=True)
    train_ind = [i for i in range(len(train_x))]
    test_ind = [j for j in range(len(test_x))]
    
    sampler = OTFSampler(index=train_ind) 
    err_list, r2_list, err_unscaled ,r2_unscaled , train_size, indlist  = list(), list(), list(), list(), list(), list()
    reduced_original,init = sampler._randsampler(width=Nseed) #reduced_original holds all train_ind except the ones added to init array. 
    [indlist.append(val) for val in init if val not in indlist ]

    print("\n..............................\n")   

    for trial in tqdm(range(1, Ntrials+1)):

        torch.manual_seed(args.randomseed) #Set the random seed for torch inside the loop
        window = random.sample(reduced_original, args.querywidth)
        x_train, x_test, y_train, y_test = return_tensor(train_x,test_x,train_y,test_y,indlist, window)

        if args.mlmethod == 'EXACT' : 
            rmse, r2, gpnoise ,predictions, likelihood, model = GPmodel(x_train,x_test, y_train,y_test,numy,n_iter=args.niter, verbose=True, GP=args.mlmethod, kernel=args.kernel)

        elif args.mlmethod == 'DKL':
            if trial == 1 :
                print("\n Starting set up for {}".format(args.mlmethod) )
                if  os.path.isfile('hyperparameters.json'):
                    param = json.load(open('hyperparameters.json'))
                    print('\nReading hyperparameters : ',param)
                else:
                    raise Exception("**ERROR** DKL hyperparameter file not found! File must be provided!")
            rmse, r2, gpnoise ,predictions, likelihood, model = DKLmodel(x_train, x_test, y_train, y_test, numy=1, n_iter=int(param["n_iter"]),\
            verbose=True, GP=args.mlmethod, kernel=args.kernel, lr=float(param["lr"]),\
            l1out=int(param["l1out"]), l2out=int(param["l2out"]), l3out=int(param["l3out"]), final=int(param["final"]))       
 
        print('\nIteration:{}, RMSE: {}, r2: {}, Train size: {}, Noise : {}'.format(trial,rmse,r2,x_train.shape[0], gpnoise))
        err_list.append(rmse) ; r2_list.append(r2) 
        print("\n Inverted noise: ",scaleY.inverse_transform( np.asarray(gpnoise).reshape((1,)) ) )

        if args.strategy == 'uncertainty':
            index = unc_sampler(predictions, gpnoise, window)
        elif args.strategy == 'emoc':
            if torch.cuda.is_available():
                index = EMOC_GPU(predictions, model, gpnoise, window, x_train.shape[0], x_test.shape[0], norm=1).compute_emoc()
            else:
                index = EMOC(predictions, model, gpnoise, window, x_train.shape[0], x_test.shape[0], norm=1).compute_emoc()
        elif args.strategy == 'random':
            index = random.sample(window, 1)        

        [indlist.append(val) for val in index if val not in indlist ]
        reduced_original.remove(index[0]) #Remove the index that is going to be added to model training set
        
        rmse, r2, _ = InferGPModel(likelihood, model, test_x, test_y, scaleY)

        print("\nValidating on test set and rescaling to normal units!\n")  
        print('\nIteration:{}, RMSE: {}, r2: {}, Test size: {}\n'.format(trial, rmse, r2, test_x.shape[0]))
        err_unscaled.append(rmse) ; r2_unscaled.append(r2) ; train_size.append(x_train.shape[0] )

    _ = plot_metric_write(train_size, err_list, r2_list, err_unscaled ,r2_unscaled ,indlist)


    print("\n..............................\n") 


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
