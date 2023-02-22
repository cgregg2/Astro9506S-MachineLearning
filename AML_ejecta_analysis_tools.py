# Useful functions for analyzing ejecta data with machine learning

def set_data(non_close_ejecta_df, close_ejecta_df, reduction_type, scalar, n_scale=10):
    """
    Set the data and target for the machine learning algorithm.

    Parameters
    ----------
    non_close_ejecta_df : pandas dataframe
        Dataframe containing the non-close ejecta data.
    close_ejecta_df : pandas dataframe
        Dataframe containing the close ejecta data.
    reduction_type : str
        Type of reduction to perform on the non-close ejecta data. Options are:
        'equalize' - reduce the number of non-close ejecta to the same number as close ejecta
        'reduce' - reduce the number of non-close ejecta to n_scale*the number of close ejecta
        'none'/'full' - do not reduce the number of non-close ejecta
    scalar : str
        Type of standarizing scalar to use. Options are:
        'stnd' - StandardScaler
        'minmax' - MinMaxScaler
        'none' - do not standardize the data
    n_scale : int (default=10)
        Number to scale the number of non-close ejecta by when reduction_type='reduce'.

    Returns
    -------
    data : numpy array
        Data to be used for machine learning.
    target : numpy array
        Target to be used for machine learning.
    """
    import pandas as pd

    # Find the number of close and full ejecta
    n_close = len(close_ejecta_df)
    n_full = len(non_close_ejecta_df)
    # Find the difference in the number of close and full ejecta
    n_diff = n_full - n_close

    if reduction_type == 'equalize':
        # Randomly select the same number of full ejecta as close ejecta
        non_close_ejecta_df = non_close_ejecta_df.sample(n=n_close)

        save_file_add = '_EQUALIZEDdata'

    elif reduction_type == 'reduce':
        # Randomly select n_scale*the number of close ejecta as full ejecta
        non_close_ejecta_df = non_close_ejecta_df.sample(n=n_scale*n_close)

        save_file_add = '_REDUCEDnonCloseData'

    elif reduction_type == 'none' or reduction_type == 'full':
        
        save_file_add = '_FULLdata'

    else:
        raise ValueError('reduction_type must be "equalize", "reduce", "none", or "full"')
        exit()

    # concatanate the dataframes
    concatenated_new = pd.concat([non_close_ejecta_df.assign(dataset=0), close_ejecta_df.assign(dataset=1)])
    
    # header of the dataframe
    df_header = concatenated_new.columns.values.tolist()

    # Split the data into data and target
    data = concatenated_new.values.T[df_header.index('host_velx'):-1].astype(float)
    data = data.T
    target = concatenated_new.values.T[-1].astype(float)

    # Standardize the data
    if scalar == 'stnd':

        # Standard scalar the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        save_file_add += '_StandardScaler'

    elif scalar == 'minmax':

        # MinMax scalar the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        save_file_add += '_MinMaxScaler'

    elif scalar == 'none':
        pass

    else:
        raise ValueError('scalar must be "stnd", "minmax", or "none"')
        exit()

    # Return the data, target and save file addition
    return data, target, save_file_add




def KernelBoundaryLine(kernel, algo, algo_name, x_train, x_test, y_train, y_test):
    """
    Plot the boundary line of the machine learning algorithm.
    https://www.kaggle.com/code/jsultan/visualizing-classifier-boundaries-using-kernel-pca

    Parameters
    ----------
    kernel : str
        Kernel to use for the Kernel PCA.
    algo : sklearn algorithm
        Machine learning algorithm to use.
    algo_name : str
        Name of the machine learning algorithm.
    x_train : numpy array
        Training data.
    x_test : numpy array
        Testing data.
    y_train : numpy array
        Training target.
    y_test : numpy array
        Testing target.

    Returns
    -------
    None.

    """

    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.decomposition import KernelPCA

    # Reduce the data to 2 dimensions for plotting with specified kernel
    reduction = KernelPCA(n_components=2, kernel = kernel)

    # Fit the data
    x_train_reduced = reduction.fit_transform(x_train)
    x_test_reduced = reduction.transform(x_test)
    
    # Fit the data to the machine learning algorithm
    classifier = algo
    classifier.fit(x_train_reduced, y_train)
    
    # Predict the data
    y_pred = classifier.predict(x_test_reduced)
    

    #Boundary Line
    X_set, y_set = np.concatenate([x_train_reduced, x_test_reduced], axis = 0), np.concatenate([y_train, y_test], axis = 0)
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)

    plt.xticks(fontsize = 3)
    plt.yticks(fontsize = 3)



def PredictKernelAccuracy(kernel, algo, algo_name, x_train, x_test, y_train, y_test, fulldata, fulltarget,  n_components = 2, data_reduced = False):
    """ 
    Predict the accuracy of the machine learning algorithm.
    https://www.kaggle.com/code/jsultan/visualizing-classifier-boundaries-using-kernel-pca

    Parameters
    ----------
    kernel : str
        Kernel to use for the Kernel PCA.
    algo : sklearn algorithm
        Machine learning algorithm to use.
    algo_name : str
        Name of the machine learning algorithm.
    n_components : int, default 2
        Number of components to use for the Kernel PCA.
    x_train : numpy array
        Training data.
    x_test : numpy array
        Testing data.
    y_train : numpy array
        Training target.
    y_test : numpy array
        Testing target.
    fulldata : numpy array
        Full set of data. (if data_reduced is True, full data set will be predicted)
    fulltarget : numpy array
        Full set of target. (if data_reduced is True, full data set will be predicted)
    data_reduced : bool, default False
        If True, the data set is a reduced set so the full data set will be predicted as well.
    
    Returns
    -------
    None.
    

    """

    # import libraries
    from sklearn.decomposition import KernelPCA
    from sklearn.metrics import classification_report, confusion_matrix

    # Reduce the data to n dimensionswith specified kernel
    reduction = KernelPCA(n_components=n_components, kernel = kernel)

    # Fit the data
    x_train_reduced = reduction.fit_transform(x_train)
    x_test_reduced = reduction.transform(x_test)
    
    # Fit the data to the machine learning algorithm
    classifier = algo
    classifier.fit(x_train_reduced, y_train)
    
    # Predict the data
    y_pred = classifier.predict(x_test_reduced)
    
    print(algo_name, kernel)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # If the data is reduced, predict the full data set
    if data_reduced:

        # Predict the full data set
        y_pred = classifier.predict(reduction.transform(fulldata))

        # Print the results
        print(algo_name, kernel, 'Full Data Set')
        print(confusion_matrix(fulltarget, y_pred))
        print(classification_report(fulltarget, y_pred))





