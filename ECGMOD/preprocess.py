"""
@author: gsivaraman@anl.gov
"""
def prepare_data(jsonfile,NormalizeY=False,NormalizeX=False):
    """
    Input Args:
    jsonfile  : JSON configuration filename (str) 
    NormalizeY : default : False (Bool)
    NormalizeX : default : False (Bool)
    dataset    : default : 'SciAdv' (Str)
    return:
    train_x : train descriptor array (array)
    train_y : train target array (array)
    test_x  : test descriptor array (array)
    test_y  : test target array (array)
    numy : multi-task number(int)
    scalerY : None if not normalized
    scalerX : None if not normalized
    """
    import json
    import pandas as pd
    import numpy as np  
    from sklearn.preprocessing import StandardScaler
    #df = pd.read_csv(filename, sep='\t',header=None)
    #df_train =  df[:8500]          #Total dataset size is 2000. Take a subset
    #df_test =   df[-1000:]
    #train_x, train_y  = df_train.values[:,:-7].astype('float32'), df_train.values[:,-5].astype('float32') # df_train.values[:,-5:-3].astype('float32')
    #test_x, test_y = df_test.values[:,:-7].astype('float32'), df_test.values[:,-5].astype('float32') # df_test.values[:,-5:-3].astype('float32')

    config_json = json.load(open(jsonfile))
    filename = config_json["csv_name"]
    train_range = config_json["train_range"] #A list with beginning and end indices.
    test_range = config_json["test_range"]      
    x_slice = config_json["x_slice"] 
    y_ind   = config_json["y_slice"] ## Assuming single task learning. Hence just an index.

    df = np.loadtxt(filename).astype('float32')    
    df_train =  df[train_range[0] : train_range[1] ]  
    df_test =   df[test_range[0] : test_range[1] ] 
    train_x, train_y  = df_train[:, x_slice[0] : x_slice[1] ], df_train[:, y_ind]
    test_x, test_y    = df_test[:, x_slice[0] : x_slice[1] ], df_test[:, y_ind]
 
    if len(test_y.shape) == 1 :
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        numy = 1
    else:
        numy = test_y.shape[1]
        
    scalerY = None
    scalerX = None
    if  NormalizeY:
        scalerY = StandardScaler()
        train_y  = scalerY.fit_transform( train_y )
        test_y = scalerY.transform(test_y)

    if NormalizeX:
        scalerX = StandardScaler()
        train_x  = scalerX.fit_transform( train_x )
        test_x = scalerX.transform(test_x)
    
    return train_x, train_y, test_x, test_y, numy, scalerY, scalerX


def return_tensor(train_x,test_x,train_y,test_y,otf_index,val_index=None):
    """
    A method to generate training/inference data based on on-the-fly indxes
    Input Args:
    train_x : Full  Train array (numpy)
    test_x : Full Test array (numpy)
    train_y : Train label (tensor)
    test_y : Test array (tensor)
    otf_index : (list)
    val_index : (list), optional (Default : None)
    Return:
    x_train, x_test, y_train, y_test :  (tensor)
    """
    import torch 
    import random
    import numpy as np 
    
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    activate_test_dataset = False    

    test_ind = [j for j in range(len(test_x))]
    if val_index == None:
        activate_test_dataset = True
        if  len(otf_index) / len(test_ind)  <= 1.0 :
            val_index = random.sample(test_ind, len(otf_index))
        else:
            val_index = test_ind
    
    x_train = np.zeros((len(otf_index),train_x.shape[1])) 
    y_train =  np.zeros((len(otf_index),train_y.shape[1]))
    x_test = np.zeros((len(val_index),train_x.shape[1]))
    y_test = np.zeros((len(val_index),train_y.shape[1]))
    
    for ind, otf in enumerate(otf_index):
        
        x_train[ind] = train_x[otf]
        y_train[ind] = train_y[otf]
        
    for ind, val in enumerate(val_index):
        if activate_test_dataset:
            x_test[ind] = test_x[val]
            y_test[ind] = test_y[val]
        else:
            x_test[ind] = train_x[val]
            y_test[ind] = train_y[val]
        
    x_train = torch.from_numpy(x_train).type(dtype)
    x_test  = torch.from_numpy(x_test).type(dtype)
    y_train  = torch.from_numpy(y_train).type(dtype)
    y_test  = torch.from_numpy(y_test).type(dtype)
    
    if torch.cuda.is_available():
        x_train, x_test, y_train, y_test = x_train.cuda(), x_test.cuda(), y_train.cuda(), y_test.cuda()

    # https://stackoverflow.com/questions/52317407/pytorch-autograd-grad-can-be-implicitly-created-only-for-scalar-outputs
    if y_train.shape[1] == 1 :  ##Bug fix for ExactGP
        y_train = torch.flatten(y_train)
        y_test = torch.flatten(y_test)
    
    return x_train, x_test, y_train, y_test
