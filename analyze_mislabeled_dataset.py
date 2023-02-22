# This script analyzes the mislabeled dataset from chosen classifier and plots the minimum heliocentric distance of the mislabeled ejecta.

# Import modules
import pickle
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
# interstellarTransforms.py (this holds import functions such as unitConversion) is located
from interstellarTransforms import unitConversions
import Firenze_read

# Creating a callable unitConversion object
uCon = unitConversions()

# File that contains the mislabeled data
pick_file = "Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF_REDUCEDnonClosedata_MinMaxScaler_MLPClassifier_mislabeled_data.pickle"

# Gaia_DR3_4472832130942575872 (nearest Gaia star) ejected 2000 particles every 1Myr from -100 Myr moved forward 110Myr (t=-100Myr --> t=10myr) simulation data
fullSimFile_pickle_dump = "integration_outputs/titan/20230208/2023-02-08_17h-26m-14s_dataDump_RKF.pickle"
fullSimFile_pick = "integration_outputs/titan/20230208/2023-02-08_17h-26m-14s_integrateMW_RKF_Executed_120561s.pickle"

# Set to True if the minimum heliocentric distances have already been calculated (not the first run through)
# If this is false, it will calculate the minimum heliocentric distances and save them to a the pickle file given here
# If this is true, it will load the minimum heliocentric distances from the pickle file given here
min_dist_calc_done = True
min_dist_calc_file = "minimum_helio_dist_mislabeled_ejecta_REDUCED.pickle"

# File base name for plots to be saved under
plots_file = min_dist_calc_file.replace('minimum_helio_dist_mislabeled_ejecta', 'minimum_helio_dist_mislabeled_ejecta_plots')

# if the minimum heliocentric distances have not been calculated, calculate them and save them to a pickle file
if min_dist_calc_done == False:

    # Load the mislabeled data
    with open(pick_file, 'rb') as f:
        mislab_df = pickle.load(f)

    # Collect names of ejecta
    ejecta_names = mislab_df['ejecta_name'].values

    # Read in close approach data from pickled file of full simulation
    dataFile = open(fullSimFile_pick, 'rb')
    data = pickle.load(dataFile)
    dataFile.close()

    # Find all minimum heliocentric distances
    min_dists = []
    for body in ejecta_names:
        # Collect the data for this body
        body = Firenze_read.extract_table(data, 'body', body)
        # Collect all heliocentric distances
        helio_dist = body['data']['helio_dist']
        # Find minimum heliocentric distance
        min_helio_dist = np.min(helio_dist)
        # Append to list
        min_dists.append(min_helio_dist)

    # Save the minimum heliocentric distances as a pickle file
    with open(min_dist_calc_file, 'wb') as f:
        pickle.dump(min_dists, f)

elif min_dist_calc_done == True:
    # Load the minimum heliocentric distances
    with open(min_dist_calc_file, 'rb') as f:
        min_dists = pickle.load(f)

# convert AU to pc
min_dists = np.array(min_dists)/uCon.pc2AU
print('Number of mislabeled ejecta in dataset:', len(min_dists))

# Plot the minimum heliocentric distances
plt.figure(figsize=(16,9))

# create bins list from 0 to 100 in steps of 2
bins = np.arange(0, 100, 2)

plt.hist(min_dists, bins=bins)

# vartical line at 100000AU (this is the threshold for a flagged close approach in data)
pcval = 100000/uCon.pc2AU
plt.axvline(x=pcval, color='r', linestyle='--')

# add text to the plot stating the value of 100000AU or "0.4848 pc"
plt.text(pcval-1.5, 250, '100,000 AU', rotation=90, size=12)
plt.text(pcval+0.25, 250, '0.485 pc', rotation=90, size=12)

plt.xlabel('Minimum heliocentric distance [pc]')
plt.ylabel('Number of ejecta from false positives')
# plt.savefig(plots_file+'long.png', dpi=600)

plt.clf()
plt.close()

# Plot the minimum heliocentric distances
plt.figure(figsize=(16,9))

# create bins list from 0 to 20 in steps of 0.5
bins = np.arange(0, 20, 0.5)
plt.hist(min_dists, bins=bins)

# vartical line at 100000AU (this is the threshold for a flagged close approach in data)
pcval = 100000/uCon.pc2AU
plt.axvline(x=pcval, color='r', linestyle='--')

# add text to the plot stating the value of 100000AU or "0.4848 pc"
plt.text(pcval-0.35, 250, '100,000 AU', rotation=90, size=12)
plt.text(pcval+0.1, 250, '0.485 pc', rotation=90, size=12)

plt.xlabel('Minimum heliocentric distance [pc]')
plt.ylabel('Number of ejecta from false positives')
plt.savefig(plots_file+'.png', dpi=600)
plt.clf()
plt.close()
