# Scipt that uses the collect_ejecta_params_from_file.py function to collect the necessary data from the simulation 
# output files and then creates a pandas dataframe with the data. The dataframe is then pickled and saved to a file.

# Importing modules
import numpy as np
import pickle
from collect_ejecta_params_from_file import collect_ejecta_params_from_file as cef
import pandas as pd


############# Simulation output files that are used to collect the necesasry data: #############

# Gaia_DR3_4472832130942575872 (nearest Gaia star) ejected 2000 particles every 1Myr from -100 Myr moved forward 110Myr (t=-100Myr --> t=10myr)
fullSimFile_pickle_dump = "integration_outputs/titan/20230208/2023-02-08_17h-26m-14s_dataDump_RKF.pickle"
fullSimFile_pick = "integration_outputs/titan/20230208/2023-02-08_17h-26m-14s_integrateMW_RKF_Executed_120561s.pickle"
# (ONLY CLOSE APPROACHES FROM ABOVE SIMULATION)
close_approach_pickle_dump =    "integration_outputs/titan/20230208/CloseApproaches/17h-26m-14s/20230213/2023-02-13_09h-45m-25s_dataDump_RKF.pickle"
closeSimFile_pick ="integration_outputs/titan/20230208/CloseApproaches/17h-26m-14s/20230213/2023-02-13_09h-45m-25s_integrateMW_RKF_Executed_406s.pickle"

# Save output file name
output_file_name = 'Gaia_DR3_4472832130942575872_2000x1Myr_-100Myr_110Myr_ejectaDF.pickle'
################################################################################################

# collecting the close approach ejecta data from the simulation output files using the collect_ejecta_params_from_file.py fucntion
close_ejecta_df = cef(closeSimFile_pick, close_approach_pickle_dump, 'close approach bodies.')

# Doing the same thing for ALL the ejecta in the simulation
full_ejecta_df = cef(fullSimFile_pick, fullSimFile_pickle_dump, 'full simulation bodies.')


# Erasing overlap between the two dataframes by deleting the close approach bodies from the full simulation dataframe
print('Deleting', len(close_ejecta_df['ejecta_name']), 'close approach bodies from the full simulation dataframe.')
for i in range(len(close_ejecta_df['ejecta_name'])):
	full_ejecta_df.drop(full_ejecta_df[full_ejecta_df['ejecta_name'] == close_ejecta_df['ejecta_name'].iloc[i]].index, inplace=True) # I love GitHub Copilot :)


print(len(full_ejecta_df['ejecta_name']), 'full simulation bodies in the dataframe after deleting the close approach bodies.\n')


# Saving the dataframes as pickle files
with open(output_file_name, 'wb') as f:
	pickle.dump([full_ejecta_df, close_ejecta_df], f)

print('Completed. Dataframes saved in', output_file_name)