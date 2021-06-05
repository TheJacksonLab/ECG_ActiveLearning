def multiT_data_loader(jsonfile, NormalizeY=False, NormalizeX=False):
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
    import json, os
    import pandas as pd
    import numpy as np  
    from sklearn.preprocessing import StandardScaler
    
    config_json = json.load(open(jsonfile))
    #config_json = jsonfile
    path = config_json['path']
    train_file = config_json["train_file"]
    test_file = config_json["test_file"]
    x_slice = config_json["x_slice"] 
    y_ind   = config_json["y_slice"] ## Assuming single task learning. Hence just an index.
    
    df_train =  np.load( os.path.join(path,train_file ) ).astype('float32')    
    df_test =   np.load(os.path.join(path,test_file ) ).astype('float32')
    
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
