# Function that collects the ejecta parameters from the given file
def collect_ejecta_params_from_file(simfile_pickled, pickled_dump, stringWithEjectaType):
	"""
	Collects the ejecta parameters from the given files and creates a pandas dataframe.

	Parameters
	----------
	simfile_pickled : str
		The path to the pickled simulation file with all outputs.
	pickled_dump : str
		The path to the pickled dump file with the final simulation point that contains all simulation parameters.
	stringWithEjectaType : str
		A string that describes the type of ejecta that is being collected. For example, 'close approach bodies' or 'full simulation bodies'.

	Returns
	-------
	ejecta_dict : dataframe
		A pandas dataframe containing the ejecta parameters.

	"""

	# Importing modules
	import pickle
	import numpy as np
	import sys
	import pandas as pd

	# adding GalacticSimulation folder to the system path. This is where interstellarTransforms.py 
	# (this holds import functions such as unitConversion) is located
	sys.path.insert(0, "L:/GitHub/PhD_Work/InterstellarMeteoroids/GalacticSimulation")
	from interstellarTransforms import unitConversions

	# Creating a callable unitConversion object
	uCon = unitConversions()

	print('\nLooking at', stringWithEjectaType)

	# Read in data from pickled file
	dataFile = open(simfile_pickled, 'rb')
	data = pickle.load(dataFile)
	dataFile.close()

	# How big is this data set
	print('length of data:', data['info']['len'])

	# This is the data header
	header = data['tops']
	print("File header: \n", header, '\n')

	# How many bodies are in this simulation
	bodys_in_file = data['data']['body']
	A_set = set(bodys_in_file)
	names=list(A_set)

	# How many bodies are ejecta?
	# Eliminate sun and host star from names (only objects with "debris" in their name are ejecta)
	names = [i for i in names if "debris" in i]
	num_bods = len(names)
	print(num_bods, stringWithEjectaType)

	# Looking at simulation data that was pickled. This contains simulation parameters and list
	# 	of GalacticCoord objects in simulation
	dataFile = open(pickled_dump, 'rb')
	bulge,disk,halo,potent,method,tol,dt,ejection,closeApproach,bodies = pickle.load(dataFile)
	dataFile.close()


	# Looking at the ejection data, each body has an initialEject attribute which is a list of the ejection parameters.
	# The ejection parameters are: [host, vel, delta_v, JD, ejection_param, pos]
	# host = the name of the host star
	# vel = the velocity of the host star
	# delta_v = the velocity of the ejecta relative to the host star
	# JD = the Julian Date of the ejection
	# ejection_param = the ejection parameter of the ejecta (0 if they will not eject particles in the simulation, 1 if they will) These should all be 0 here
	# pos = the position of host star

	# Creating a header for these ejection parameters
	eject_head = ['host', 'vel', 'delta_v','JD', 'ejection_param', 'pos']
	df_header = ['ejecta_name', 'host', 'host_velx', 'host_vely', 'host_velz', 'host_vel_mag', 'eject_velx', 'eject_vely', 'eject_velz', 'eject_vel_mag', 'eject_ang', 'eject_ang_deg', 'JD', 'posx', 'posy', 'posz', 'pos_mag']

	# Creating a list to hold the ejection parameters
	ejection_params = []

	print('Finding ejection params of ' + stringWithEjectaType +'..')
	for body in bodies:
		new_body = []
		# If the body is an ejecta body, then calculate the ejection velocity and angle
		if body.initialEject != [] and body.initialEject != 'none':

			# Collecting the ejection parameters
			eject_para = body.initialEject

			# Creating a list to hold the ejection parameters for this body
			new_body = [body.name, eject_para[eject_head.index('host')]]

			# Collecting host velocity vector and converting to km/s
			host_vel = np.array(eject_para[eject_head.index('vel')])*uCon.AUyr2kms
			norm_host_vel = np.linalg.norm(host_vel)

			# adding the host velocity components to the list
			for element in host_vel:
				new_body.append(element)

			# Adding the magnitude of the host velocity to the list
			new_body.append(norm_host_vel)

			# Collecting ejection velocity vector and converting to km/s
			ejection_vel = np.array(eject_para[eject_head.index('delta_v')])*uCon.AUyr2kms
			norm_ejection_vel = np.linalg.norm(ejection_vel)

			# adding the ejection velocity components to the list
			for element in ejection_vel:
				new_body.append(element)

			# Adding the magnitude of the ejection velocity to the list
			new_body.append(norm_ejection_vel)

			# Calculating the ejection angle and adding it to the list
			ejection_angle = np.dot(ejection_vel,host_vel)/(norm_ejection_vel*norm_host_vel)
			new_body.append(ejection_angle)
			# Converting the ejection angle to degrees and adding it to the list
			new_body.append(np.degrees(np.arccos(ejection_angle)))

			# Adding the Julian Date of the ejection to the list
			new_body.append(eject_para[eject_head.index('JD')])

			# Collecting the position vector of the host star
			host_pos = np.array(eject_para[eject_head.index('pos')])
			norm_host_pos = np.linalg.norm(host_pos)

			# adding the host position components to the list
			for element in host_pos:
				new_body.append(element)
			
			# Adding the magnitude of the host position to the list
			new_body.append(norm_host_pos)

		# Adding the new body to the list of ejection parameters if it is an ejecta
		if new_body != []:
			ejection_params.append(new_body)

	# Clearing memory
	bulge,disk,halo,potent,method,tol,dt,ejection,closeApproach,close_bodies = [],[],[],[],[],[],[],[],[],[]


	# Creating a pandas dataframe to hold the ejection parameters
	eject_frame = pd.DataFrame(ejection_params, columns=df_header)

	print('Done finding ejection params of', stringWithEjectaType, 'Output as dataframe.\n')

	return eject_frame