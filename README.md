# Astro9506S-MachineLearning
Machine learning project related to interstellar meteoroids for the purpose of Astronomy 9506S.

The results of a galactic simulation where the nearest star in the Gaia catalogue ejected 2000 particles every 1 million years (Myr) beginning 100 Myrs ago and ending 10 Myr in the future (total duration of 110 Myr) were investigated to determine which parameters of the ejection determine whether the material will have a close approach with the Solar System. Of the 220,000 total ejections, 152 have a close approach. There is no clear distinguishing characteristic of close approache material compared to other ejecta. Here, machine learning techniques are put to use to attempt to distinguish between these two outcomes.

Code order of use:

1 - preparing_ejecta_analysis.py

    This uses collect_ejecta_params_from_file.py to store useful data in dataframe
    
2 - analyze_pickled_ejecta_dataframes.py

    This is where the machine learning techniques can be put to use
