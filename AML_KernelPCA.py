# This is a scipt that will perform a variety of Kernel PCA on the data and show a plot of the results and accuracy.
# Remember to set the parameters at the top of the script.

# Importing modules
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import AML_ejecta_analysis_tools as AML_tools

from sklearn.decomposition import KernelPCA
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Start time of counter
start_time = time.perf_counter()

# This is the file that contains the pickled dataframes
df_pickle_file = 'L:/GitHub/PhD_Work/InterstellarMeteoroids/GalacticSimulation/plotting/Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF.pickle'

# This is the root name of the plots that will be created
plot_file_BASEname = 'L:/GitHub/PhD_Work/InterstellarMeteoroids/GalacticSimulation/plotting/Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF'


###################################################################################################
###################################################################################################
###################################################################################################

# This is where you set the parameters for the analysis

# dataset reduction choices
data_reduction_type = 'reduce' # 'reduce', 'equalize', or 'none'
                               # 'equalize' the number of close and non-close ejecta
                               # 'reduce' the number of non-close ejecta to n*number of close ejecta
n_scale = 10 # Number of non-close ejecta to scale to if 'reduce' is True 

# Scalar selection
scalar = 'minmax' # 'stnd', 'minmax', or 'none
                  # 'stnd' - StandardScaler
                    # 'minmax' - MinMaxScaler

# Kernel PCA parameters
n_components = 7 # Number of components to keep

###################################################################################################
###################################################################################################
###################################################################################################


# Load the pickled dataframes
with open(df_pickle_file, 'rb') as f:
    non_close_ejecta_df, close_ejecta_df = pickle.load(f)


# Create a concatenated dataframe
concatenated = pd.concat([non_close_ejecta_df.assign(dataset=0), close_ejecta_df.assign(dataset=1)])
# Recall header =>
df_header = concatenated.columns.values.tolist()
print('header: ', df_header)

# Create the data and target arrays with equalize_data function
data, target, save_file_add = AML_tools.set_data(non_close_ejecta_df, close_ejecta_df, data_reduction_type, scalar=scalar, n_scale=n_scale)

# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)


# Plotting each kernels first 2 components and the decision boundary of the classifier
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Classifiers and Kernel PCA')


#Logistic Regression   

ax = plt.subplot(7,5,1)
ax.set_title('Linear PCA')
ax.set_ylabel('Logistic \n Regression', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test)

ax = plt.subplot(7,5,2)
ax.set_title('RBF PCA')
AML_tools.KernelBoundaryLine('rbf', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test)

ax = plt.subplot(7,5,3)
ax.set_title('Poly PCA')
AML_tools.KernelBoundaryLine('poly', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test)

ax = plt.subplot(7,5,4)
ax.set_title('Sigmoid PCA')
AML_tools.KernelBoundaryLine('sigmoid', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test)

ax = plt.subplot(7,5,5)
ax.set_title('Cosine PCA')
AML_tools.KernelBoundaryLine('cosine', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test)


#Naive Bayes
ax = plt.subplot(7,5,6)
ax.set_ylabel('Naive \n Bayes', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,7)
AML_tools.KernelBoundaryLine('rbf', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,8)
AML_tools.KernelBoundaryLine('poly', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,9)
AML_tools.KernelBoundaryLine('sigmoid', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,10)
AML_tools.KernelBoundaryLine('cosine', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test)

#K-Nearest Neighbors
ax = plt.subplot(7,5,11)
ax.set_ylabel('KNN', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,12)
AML_tools.KernelBoundaryLine('rbf', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,13)
AML_tools.KernelBoundaryLine('poly', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,14)
AML_tools.KernelBoundaryLine('sigmoid', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,15)
AML_tools.KernelBoundaryLine('cosine', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test)

#Random Forest
ax = plt.subplot(7,5,16)
ax.set_ylabel('Random \n Forest', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,17)
AML_tools.KernelBoundaryLine('rbf', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,18)
AML_tools.KernelBoundaryLine('poly', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,19)
AML_tools.KernelBoundaryLine('sigmoid', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,20)
AML_tools.KernelBoundaryLine('cosine', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test)

#Support Vector - linear
ax = plt.subplot(7,5,21)
ax.set_ylabel('SVM \n Linear', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,22)
AML_tools.KernelBoundaryLine('rbf', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,23)
AML_tools.KernelBoundaryLine('poly', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,24)
AML_tools.KernelBoundaryLine('sigmoid', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,25)
AML_tools.KernelBoundaryLine('cosine', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test)

#Support Vector - RBF
ax = plt.subplot(7,5,26)
ax.set_ylabel('SVM \n rbf', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,27)
AML_tools.KernelBoundaryLine('rbf', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,28)
AML_tools.KernelBoundaryLine('poly', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,29)
AML_tools.KernelBoundaryLine('sigmoid', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,30)
AML_tools.KernelBoundaryLine('cosine', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test)


#Support Vector - Poly
ax = plt.subplot(7,5,31)
ax.set_ylabel('SVM \n poly', rotation = 0, labelpad=30, fontsize = 10)
AML_tools.KernelBoundaryLine('linear', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,32)
AML_tools.KernelBoundaryLine('rbf', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,33)
AML_tools.KernelBoundaryLine('poly', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,34)
AML_tools.KernelBoundaryLine('sigmoid', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test)
ax = plt.subplot(7,5,35)
AML_tools.KernelBoundaryLine('cosine', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test)

# save
save_file = plot_file_BASEname + save_file_add + '_ClassifiersAndKernelPCA.png'
fig.savefig(save_file, dpi=600, bbox_inches='tight')


#  Now testing the accuracy of the models

if data_reduction_type in ['reduce', 'equalize']:
    data_reduced = True
else:
    data_reduced = False

# Create the data and target arrays for full prediction
fulldata = concatenated.values.T[df_header.index('host_velx'):-1].astype(float)
fulldata = fulldata.T
fulltarget = concatenated.values.T[-1].astype(float)

#Logistic Regression   
AML_tools.PredictKernelAccuracy('linear', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', LogisticRegression(), "Logistic Regression", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)


#Naive Bayes
AML_tools.PredictKernelAccuracy('linear', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', GaussianNB(), "Naive Bayes", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)

#K-Nearest Neighbors
AML_tools.PredictKernelAccuracy('linear', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', KNeighborsClassifier(), "KNN", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)

#Random Forest
AML_tools.PredictKernelAccuracy('linear', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', RandomForestClassifier(), "Random Forest", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)

#Support Vector - linear
AML_tools.PredictKernelAccuracy('linear', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', SVC(kernel = 'linear'), "SVM - Linear", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)

#Support Vector - RBF
AML_tools.PredictKernelAccuracy('linear', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', SVC(kernel = 'rbf'), "SVM - rbf", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)


# Support Vector - Poly
AML_tools.PredictKernelAccuracy('linear', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('rbf', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('poly', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('sigmoid', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)
AML_tools.PredictKernelAccuracy('cosine', SVC(kernel = 'poly'), "SVM - poly", x_train, x_test, y_train, y_test, fulldata, fulltarget, n_components=n_components, data_reduced=data_reduced)