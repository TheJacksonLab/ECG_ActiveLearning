def plot_metric_write(train_size,err_list,r2_list, err_unscaled ,r2_unscaled , indlist, tag=None):
    '''
    Plot the metric/ write evolution history  over the trial using this function
    :param train_size: (list)
    :param err_list:  (list)
    :param r2_list:  (list)
    :param err_unscaled : (list) Error rescaled to energy scale
    :param r2_unscaled : (list)
    :param indlist: (list) 
    :param tag: (str) String to tag to output file name 
    Return type: figure object
    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    plt.rcParams["font.family"] = "Arial"
    mpl.style.use('seaborn')

    fig = plt.figure(figsize=(18.5,10.5),dpi=300)
    ax1 = fig.add_subplot(211)
    ax1.plot(train_size,err_list,'*--',color='r',lw=3.5)
    ax1.set_ylabel('RMSE',fontsize=24)
    ax1.set_xlabel('AL sample size',fontsize=24)
    ax1.set_title("AL Evolution",fontsize=26)
    ax1.tick_params(axis ='both', which ='major',
               labelsize = 20)

    ax1.tick_params(axis ='both', which ='minor',  labelsize = 20)
    ax1.set_ylim([min(err_list), max(err_list)])
    ax1.set_xlim([train_size[0]-1, max(train_size)])

    ax2 = fig.add_subplot(212)
    ax2.plot(train_size,r2_list,'^--',color='g',lw=3.5)
    ax2.set_ylabel(r'r$^{2}$',fontsize=24)
    ax2.set_xlabel('AL sample size',fontsize=24)
    ax2.tick_params(axis ='both', which ='major',  labelsize = 20)
    ax2.tick_params(axis ='both', which ='minor',  labelsize = 20)
    ax2.set_ylim([min(r2_list), max(r2_list)])
    ax2.set_xlim([train_size[0]-1, max(train_size)])

    plt.tight_layout()
    plt.draw()
    fig.savefig('AL.png',dpi=300)

    xarray = np.array(train_size)
    yarray = np.array(err_list)
    zarray = np.array(r2_list)
    array_err_unscaled = np.array(err_unscaled)
    array_r2_unscaled = np.array(r2_unscaled)

    indarray = np.array(indlist)

    data = np.array([xarray, yarray, zarray, array_err_unscaled, array_r2_unscaled])
    data = data.T

    indexarray_path = "AL_train_Array.dat"

    if tag == None:
        datafile_path = "AL_history.dat"
    elif tag != None:
        datafile_path = "AL_history_{}.dat".format(tag)

    with open(datafile_path, 'w+') as datafile_id:
        np.savetxt(datafile_id, data, fmt=['%d','%f','%f','%f','%f'])

    with open(indexarray_path, 'w+') as indexfile_id:
        np.savetxt(indexfile_id, indarray, fmt=['%d'])


    return fig
