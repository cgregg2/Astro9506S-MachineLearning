# This script is where I will analyze the pickled dataframes with machine learning techniques

# Importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import time
import AML_ejecta_analysis_tools as AML_tools

# Importing machine learning modules
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


# Start time of counter
start_time = time.perf_counter()

# This is the file that contains the pickled dataframes (From preparing_ejecta_analysis.py)
df_pickle_file = 'L:/Github/PhD_Work/InterstellarMeteoroids/GalacticSimulation/plotting/Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF.pickle'

# This is the root name of the plots that will be created
plot_file_BASEname = 'L:/Github/PhD_Work/InterstellarMeteoroids/GalacticSimulation/plotting/Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF'



###################################################################################################
###################################################################################################
###################################################################################################

# This is where you set the parameters for the analysis

# dataset plots to be created before any machine learning
create_pairplots = False # Create pairplots of the ejection parameters
create_correlation_matrix = False # Create a correlation matrix of the ejection parameters

# dataset reduction choices
data_reduction_type = 'reduce' # 'reduce', 'equalize', or 'none'
                               # 'equalize' the number of close and non-close ejecta
                               # 'reduce' the number of non-close ejecta to n*number of close ejecta
n_scale = 10 # Number of non-close ejecta to scale to if 'reduce' is True 

# Scalar selection
scalar = 'minmax' # 'stnd', 'minmax', or 'none
                  # 'stnd' - StandardScaler
                    # 'minmax' - MinMaxScaler

# Dimensionality reduction
do_pca = False
do_tSNE = False # t-distributed Stochastic Neighbor Embedding
## Subset of parameters for tSNE
n_comp = 3 # Number of components to reduce to for tSNE
perp = 250 # Perplexity for tSNE
n_iter = 5000 # Number of iterations for tSNE

### machine learning technique and params ###

# Unsupervised - Clustering
dbScan = False # Density-Based Spatial Clustering of Applications with Noise
## Subset of parameters for DBSCAN
min_samples=20
eps=0.75
n_jobs=-1

# Supervised - Classification
random_state = 42
test_size = 0.1
max_iter = 4000

svmClass = False # Support Vector Machine
## Subset of parameters for SVC
svm_type = 'linear' # can be: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
C=100
gamma='auto'

# Supervised - Neural Network
mlpClass = True
## Subset of parameters for MLPClassifier
hidden_layer = (100,90,80,70,65,60,50,45,40,35,30,25,20,15,10,5)
solver='adam'
shuffle=True
activation='relu'
learning_rate='adaptive'
learning_rate_init=0.0001 
validation_fraction=0.2


###################################################################################################
###################################################################################################
###################################################################################################


# Taking note if data is reduced for future use
if data_reduction_type in ['reduce', 'equalize']:
    data_reduced = True
else:
    data_reduced = False


# Load the pickled dataframes
with open(df_pickle_file, 'rb') as f:
    non_close_ejecta_df, close_ejecta_df = pickle.load(f)

# Create a concatenated dataframe
concatenated = pd.concat([non_close_ejecta_df.assign(dataset=0), close_ejecta_df.assign(dataset=1)])
# Recall header =>
df_header = concatenated.columns.values.tolist()
print('header: ', df_header)


if create_pairplots:

    # Plotting the distribution of the ejection parameters using seaborn pairplot

    sns.pairplot(close_ejecta_df)
    save_file = plot_file_BASEname + '_CLOSE_pairplot.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    sns.pairplot(non_close_ejecta_df)
    save_file = plot_file_BASEname + '_FULL_pairplot.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    sns.pairplot(concatenated, hue='dataset')
    save_file = plot_file_BASEname + '_FULL_CLOSE_pairplot.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()


if create_correlation_matrix:
    
    # put data into a numpy array for correlation matrix calculation then compute the correlation matrix
    data = concatenated.values.T[df_header.index('host_velx'):].astype(float)
    cov_data = np.corrcoef(data)

    # Plot the correlation matrix
    plt.figure(figsize=(16,9))
    img = plt.matshow(cov_data, cmap=plt.cm.coolwarm, vmin=-1, vmax=1, fignum=1)
    plt.colorbar(img)

    # Add the labels
    for i in range(cov_data.shape[0]):
        for j in range(cov_data.shape[1]):
            plt.text(x=j, y=i, s="{:.2f}".format(cov_data[i, j]), va='center', ha='center', color='k', size=12)

    plt.xticks(range(len(df_header[df_header.index('host_velx'):])), df_header[df_header.index('host_velx'):], rotation=90)
    plt.yticks(range(len(df_header[df_header.index('host_velx'):])), df_header[df_header.index('host_velx'):])
    save_file = plot_file_BASEname + '_FULL_CLOSE_correlation_matrix.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()


# Create the data and target arrays with set choices and set_data function
data, target, save_file_add = AML_tools.set_data(non_close_ejecta_df, close_ejecta_df, data_reduction_type, scalar, n_scale=n_scale)


if do_pca:

    save_file_add += '_PCA'

    # PCA
    dimensionReduction = PCA(n_components=len(df_header[df_header.index('host_velx'):-1]))
    dimensionReduction.fit(data)

    # Print the results
    print("\nExplained variance ratio:\n", dimensionReduction.explained_variance_ratio_)
    print("\nSum of explained variance ratio:\n", sum(dimensionReduction.explained_variance_ratio_))
    print("\nExplained variance:\n", dimensionReduction.explained_variance_)
    print("\nNoise variance:\n", dimensionReduction.noise_variance_)
    print("\nNumber of components:\n", dimensionReduction.n_components_)

    # Plot histogram of the explained variance ratio
    plt.figure(figsize=(16,9))
    plt.bar(range(1, len(dimensionReduction.explained_variance_ratio_)+1), dimensionReduction.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.xticks(range(1, len(dimensionReduction.explained_variance_ratio_)+1))
    save_file = plot_file_BASEname + save_file_add + '_explained_variance_ratio_HIST.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    # Determine how many components are needed to explain 95% of the variance
    num_components = 0
    explained_variance_ratio_sum = 0
    while explained_variance_ratio_sum < 0.95:
        num_components += 1
        explained_variance_ratio_sum += dimensionReduction.explained_variance_ratio_[num_components-1]
    print("\nNumber of components needed to explain 95% of the variance: ", num_components)
    
    # Redo PCA with the number of components needed to explain 95% of the variance
    print("\nRedoing PCA with the number of components needed to explain 95% of the variance...")
    dimensionReduction = PCA(n_components=num_components)
    dimensionReduction.fit(data)
    # Print the results
    print("\nExplained variance ratio:\n", dimensionReduction.explained_variance_ratio_)
    print("\nSum of explained variance ratio:\n", sum(dimensionReduction.explained_variance_ratio_))
    print("\nExplained variance:\n", dimensionReduction.explained_variance_)
    print("\nNoise variance:\n", dimensionReduction.noise_variance_)
    print("\nNumber of components:\n", dimensionReduction.n_components_)
    
    # Transform the data
    data_pca = dimensionReduction.transform(data)

    # put data into a numpy array for correlation matrix calculation then compute the correlation matrix
    data_pca_T = data_pca.T
    cov_data_pca = np.corrcoef(data_pca_T)

    print("Plotting the correlation matrix of PCA...")
    plt.figure(figsize=(16,9))
    img = plt.matshow(cov_data_pca, cmap=plt.cm.coolwarm, vmin=-1, vmax=1, fignum=1)
    plt.colorbar(img)

    # Add the labels
    for i in range(cov_data_pca.shape[0]):
        for j in range(cov_data_pca.shape[1]):
            plt.text(x=j, y=i, s="{:.2f}".format(cov_data_pca[i, j]), va='center', ha='center', color='k', size=12)

    save_file = plot_file_BASEname + save_file_add + '_correlation_matrix.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    # Plot the PCA components against each other using seaborn pairplot
    print("Plotting the pairplot...")
    columns = ['PC'+str(i) for i in range(1, dimensionReduction.n_components+1)] # Create the column names
    pca_df = pd.DataFrame(data_pca, columns=columns)
    pca_df['target'] = target

    sns.pairplot(pca_df, hue='target')
    save_file = plot_file_BASEname + save_file_add + '_pairplot.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    data = data_pca


elif do_tSNE:

    save_file_add += '_tSNE'

    # Lets try to use tSNE
    dimensionReduction = TSNE(n_components=n_comp, verbose=1, perplexity=perp, n_iter=n_iter)
    data_tsne = dimensionReduction.fit_transform(data)

    # put data into a numpy array for correlation matrix calculation then compute the correlation matrix
    data_tsne_T = data_tsne.T
    cov_data_tsne = np.corrcoef(data_tsne_T)

    print("Plotting the correlation matrix of t-SNE...")
    plt.figure(figsize=(16,9))
    img = plt.matshow(cov_data_tsne, cmap=plt.cm.coolwarm, vmin=-1, vmax=1, fignum=1)
    plt.colorbar(img)

    # Add the labels
    for i in range(cov_data_tsne.shape[0]):
        for j in range(cov_data_tsne.shape[1]):
            plt.text(x=j, y=i, s="{:.2f}".format(cov_data_tsne[i, j]), va='center', ha='center', color='k', size=12)

    save_file = plot_file_BASEname + save_file_add + '_correlation_matrix.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    print("Plotting the pairplot...")
    # Plot the PCA using seaborn pairplot
    columns = ['t-SNE '+str(i) for i in range(1, n_comp+1)]
    tsne_df = pd.DataFrame(data_tsne, columns=columns)
    tsne_df['target'] = target

    sns.pairplot(tsne_df, hue='target')
    save_file = plot_file_BASEname + save_file_add + '_pairplot.png'
    plt.savefig(save_file, dpi=600)
    plt.clf()
    plt.close()

    data = data_tsne


if dbScan:

    # Lets try to use DBSCAN
    save_file_add += '_DBSCAN'

    dbscan = DBSCAN(min_samples=min_samples, eps=eps, n_jobs=n_jobs)

    # Fit the DBSCAN object to the data
    dbscan.fit(data)

    # Get the labels for each data point
    labels = dbscan.labels_

    # Get the number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print('Number of clusters found with DBSCAN: ', n_clusters)


elif mlpClass:

    save_file_add += '_MLPClassifier'

    print('Training the MLPClassifier...')

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)

    # Create the model
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer, solver=solver, max_iter=max_iter, shuffle=shuffle, 
                    activation=activation, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                    validation_fraction=validation_fraction)

    # Train the model
    classifier.fit(x_train, y_train)

    # Predict the test set
    predictions = classifier.predict(x_test)

    # Print the results
    print('\nMLPClassifier train set score: ', classifier.score(x_train, y_train))
    print("\nMLPClassifier test set score: ", classifier.score(x_test, y_test))

    # Print the classification report
    print(sklearn.metrics.classification_report(y_test, predictions))

    # Print the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:\n", cm)

    # Plot the confusion matrix
    plt.figure(figsize=(16,9))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Close Approach', 'Close Approach'])
    disp.plot()
    save_file = plot_file_BASEname + save_file_add + '_confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(save_file, dpi=600, pad_inches=5)
    plt.clf()
    plt.close()

elif svmClass:

    save_file_add += '_SVC' + svm_type

    print('Training the SVC...')

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)

    # Create the model
    classifier = SVC(kernel=svm_type, C=C, gamma=gamma, max_iter=max_iter)

    # Train the model
    classifier.fit(x_train, y_train)

    # Predict the test set
    predictions = classifier.predict(x_test)

    # Print the results
    print('\nSVC train set score: ', classifier.score(x_train, y_train))
    print("\nSVC test set score: ", classifier.score(x_test, y_test))

    # Print the classification report
    print(sklearn.metrics.classification_report(y_test, predictions))

    # Print the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)
    print("\nConfusion matrix:\n", cm)

    # Plot the confusion matrix
    plt.figure(figsize=(16,9))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Close Approach', 'Close Approach'])
    disp.plot()
    save_file = plot_file_BASEname + save_file_add + '_confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(save_file, dpi=600, pad_inches=5)
    plt.clf()
    plt.close()


if data_reduced and do_tSNE==False:

    # Predict all the data now with the model trained on the data subset
    all_data, all_targets, add_save_file_str = AML_tools.set_data(non_close_ejecta_df, close_ejecta_df, 'none', scalar)
    
    # if pca or tsne, transform the data
    if do_pca:
        all_data = dimensionReduction.transform(all_data)        

    # Predict the all the data with the model trained on the data subset
    all_predictions = classifier.predict(all_data)

    # Print the classification report
    print(sklearn.metrics.classification_report(all_targets, all_predictions))

    # Print the confusion matrix
    cm = sklearn.metrics.confusion_matrix(all_targets, all_predictions)
    print("\nConfusion matrix:\n", cm)

    # Plot the confusion matrix
    plt.figure(figsize=(16,9))
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Close Approach', 'Close Approach'])
    disp.plot(values_format='')
    save_file = plot_file_BASEname + save_file_add + '_confusion_matrix_reTest_ALLDATA.png'
    plt.tight_layout()
    plt.savefig(save_file, dpi=600, pad_inches=5)
    plt.clf()
    plt.close()

    # Get the indices of the mislabeled data
    mislabeled_indices = np.where(all_predictions != all_targets)[0]

    print(len(mislabeled_indices), 'mislabeled indices out of', len(all_targets), 'total data points.')

    # Make dataframe of mislabeled data
    mislabeled_df = concatenated.iloc[mislabeled_indices]

    # Save the mislabeled data as pickle
    save_file = plot_file_BASEname + save_file_add + '_mislabeled_data.pickle'
    mislabeled_df.to_pickle(save_file)
    print('Saved mislabeled data to: ', save_file)

elif data_reduced and do_tSNE:
    print('Cannot do t-SNE transform on all the data when data was reduced/equalized for fit. Would need to refit. If this was intended, do not reduce data in beginning. Skipping...')
    pass


# Record the end time
end_time = time.perf_counter() - start_time
print("Total time: ", end_time)