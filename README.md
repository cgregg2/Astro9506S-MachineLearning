# Astro9506S-MachineLearning
Machine learning project related to interstellar meteoroids for the purpose of Astronomy 9506S.

The results of a galactic simulation where the nearest star in the Gaia catalogue ejected 2000 particles every 1 million years (Myr) beginning 100 Myrs ago and ending 10 Myr in the future (total duration of 110 Myr) were investigated to determine which parameters of the ejection determine whether the material will have a close approach with the Solar System. Of the 220,000 total ejections, 152 have a close approach (0.07%). There is no clear distinguishing characteristic of close approache material compared to other ejecta. Here, machine learning techniques are put to use to attempt to distinguish between these two outcomes.

interstellarTransforms.py is not a piece of code that was created for this project, but for my simulation. It holds a unit convserion function that I make use of here. 

Code order of use:

1 - preparing_ejecta_analysis.py

    This uses collect_ejecta_params_from_file.py to store useful data in dataframe
    
2 - AML_KernelPCA.py

    This analyzes 5 kernels and 7 classifiers to show decision boundary on 2d plot as well as outputting predicted accuracy of each classifier
    
3 - analyze_pickled_ejecta_dataframes.py

    This is where alternate machine learning techniques can be put to use
    
4 - analyze_mislabeled_dataset.py

    This is where you can take a closer look at the mislabeled data from the classifier. This creates a histogram to check the actual minimal heliocentric distances of the data said to be close approaches by the classifier that weren't flagged in the simualtion.
