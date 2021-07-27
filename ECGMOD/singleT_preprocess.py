def singleT_data_loader(jsonfile, NormalizeY=False, NormalizeX=False):
    """
    The goal here is to load global numpy files and parse single T training arrays, while retaining the multiT validator!
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
    x_slice = config_json["x_slice"] #Pick the feature column dimensions
    y_ind   = config_json["y_slice"] ## single task = index, multi-task = list 
    train_row_slice = config_json["train_refT_Xslice"] #This is where single T row width is defined as a list.


    df_train =  np.load( os.path.join(path,train_file ) ).astype('float32')    
    df_test =   np.load(os.path.join(path,test_file ) ).astype('float32')
   
    

    if type(y_ind) == list:
        train_x, train_y  = df_train[8500*train_row_slice[0] : 8500*train_row_slice[1] , x_slice[0] : x_slice[1] ], df_train[8500*train_row_slice[0] : 8500*train_row_slice[1], y_ind[0] : y_ind[1] ]
        test_x, test_y    = df_test[:, x_slice[0] : x_slice[1] ], df_test[:,  y_ind[0] : y_ind[1] ]
    elif type(y_ind) == int:
        train_x, train_y  = df_train[8500*train_row_slice[0] : 8500*train_row_slice[1], x_slice[0] : x_slice[1] ], df_train[8500*train_row_slice[0] : 8500*train_row_slice[1], y_ind]
        test_x, test_y    = df_test[:, x_slice[0] : x_slice[1] ], df_test[:, y_ind]
    
    if len(test_y.shape) == 1 :
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        numy = 1
    else:
        numy = test_y.shape[1]
    
    refT = str(set(df_train[ 8500*train_row_slice[0] : 8500*train_row_slice[1] , -1 ] ) )[1:-1]
    print("\nReading a total of  {} query samples drawn at T = {} K ".format(train_x.shape[0],refT) )
    print("\n Number of tasks to learn:", numy) 

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
