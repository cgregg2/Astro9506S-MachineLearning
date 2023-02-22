# Functions that will be able to transform GAIA data to galactocentric coords, in different ref frames
def rot(dir,ang):
	"""
	This function takes a rotation axis and an angle tp create a 
	rotation matrix that will rotate about the rotation axis by the
	angle in the clockwise direction. 

	Inputs: 
		
		dir - roation axis (x,y, or z)

		ang - angle in radians 
		

	Outputs: 

		M - rotation matrix

	Example of use:

		M = rot('x',3.14159/2)

		returns np.array([[1,0,0],[0,0,1],[0,-1,0]])
	"""
	import numpy as np

	try:
		float(ang)
	except ValueError:
		print('WARNING!: TypeError(rot) - input ang format not a float', type(ang))
		return 'NaN'

	if dir == 'x':
		M = np.array([[1,0,0],[0,np.cos(ang),np.sin(ang)],[0,-np.sin(ang),np.cos(ang)]])
	elif dir == 'y':
		M = np.array([[np.cos(ang),0,-np.sin(ang)],[0,1,0],[np.sin(ang),0,np.cos(ang)]])
	elif dir == 'z':
		M = np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
	else:
		print("WARNING!!: rot() returning 'NaN' as dir input was invalid. Value must be x,y, or z.")
		return 'NaN'

	return M




def parallax2dist(parallax, Unit_Conversion = 206264.8075):
	"""
	 This function accepts a parallax (in units of mili-arcsec) and returns a distance (default AU)

	Input: 
		parallax (integer, float, or string, units of mili-arcsec)

	Outputs: 

			d - distance (default in AU) (float), to change this, change optional variable "Unit_Conversion"
					to a value of pc_2_unit (for AU, this is 206264.8075)			

	Example of use:

		d = parallax2dist(768.066539187357)

		returns d = 

	"""

	try:
		para = float(parallax)
	except ValueError:
		print('WARNING!: TypeError(parallax2dist) - input format not a float', type(parallax))
		return 'NaN'

	# mili-arcsec to arcsec
	para = para*(10**-3)

	# d in pc is 1/para, convert to prefered unit (AU default)
	d = (1/para)*Unit_Conversion

	return d

class unitConversions:
	"""
	This is a class to save values for unit conversion

	"""

	def __init__(self):
		self.km2AU = 1/149597870.691
		self.s2yr = 1/31558196.01
		# self.s2yr = 1/31557600
		self.yr2day = 365.2568983263281
		# self.yr2day = 365.25
		self.pc2AU = 206264.8075
		self.kms2AUyr = self.km2AU/self.s2yr
		self.AUyr2kms = self.s2yr/self.km2AU
		self.km2pc = self.km2AU/self.pc2AU
		self.kms2pcyr = self.km2pc/self.s2yr
		self.J2000 = 2451544.5000000
		self.yrs2solaryr = 365.25/self.yr2day


class galacticCoord:
	"""
	This is a class for storing galactic positions and velocities. To initialize, input name of body:
			- galacticCoord(name_of_body)
	
	This class also contains functions to display and convert coordinate systems.

	Suggested units:
	epoch - year
	jd - days (not modified)
	ra - deg
	dec - deg
	para - mas
	helio_dist = AU
	v_R = km/s
	pmra = mas/yr
	pmdec = mas/yr
	ICRScart_p(v) - AU(AU/yr)
	galcart_p(v) - AU(AU/yr)
	"""

	def __init__(self, name):

		import numpy as np

		self.name = str(name)
		self.epoch = []
		self.jd = []
		self.ra = []
		self.dec = []
		self.para = []
		self.helio_dist = []
		self.helio_pos = 'none'
		self.helio_vel = 'none'
		self.v_R = []
		self.pmra = []
		self.pmdec = []
		self.ICRScart_p = []
		self.ICRScart_v = []
		self.galcart_p = []
		self.galcart_v = []
		self.Accel = np.array([0,0,0])
		self.mass = 'none'
		self.momentum = 'none'
		self.uCon = unitConversions()
		self.ejectFlag = 0
		self.CloseApproachFlag = 0
		self.initialEject ='none'

	def display_EQ(self):
		print("Coordinates for", self.name, "jd = ", self.jd,": ", "ra =", self.ra, ", dec =", self.dec)

	def display_galCart(self):
		print("Galactocentric coordinates for", self.name, "on JD = ", self.jd,": ", "Position =", self.galcart_p, ", Velocity =", self.galcart_v)

	def vR_Unit_Conversion(self,convert):
		"""
		 This function accepts the radial velcoity in km/s,AU/s, or AU/yr and converts to AU/yr,AU/s, or km/s

		Input: 

			v_R (integer, float, or string, units of km/s,AU/s, or AY/yr)

			convert (string, ['kms2AUyr', 'AUyr2kms','kms2AUs', 'AUs2kms'])

		Outputs: 
			
			v_R (float, units of AU/yr,AU/s, or km/s)	

		Example of use:

			v_R = star.vR_Unit_Conversion('kms2AUyr')

			returns v_R*km2AU/s2yr

		"""

		# Checking for type and correcting if necessary
		try:
			v_R = float(self.v_R)
		except ValueError:
			print('WARNING!: TypeError(kms2AUyr) - input format not a float', type(v_R))
			return 'NaN'
		try:
			convert = str(convert)
		except ValueError:
			print('WARNING!: TypeError(kms2AUyr) - input format not a string', type(convert))
			return 'NaN'

		if convert in ['kms2AUyr', 'AUyr2kms','kms2AUs', 'AUs2kms']:

			if convert=='kms2AUyr':
				self.v_R = self.v_R*self.uCon.km2AU/self.uCon.s2yr
			if convert=='AUyr2kms':
				self.v_R = self.v_R/self.uCon.km2AU*self.uCon.s2yr
			if convert=='kms2AUs':
				self.v_R = self.v_R*self.uCon.km2AU
			if convert=='AUs2kms':
				self.v_R = self.v_R/self.uCon.km2AU

		else:
			print('WARNING!: conversionError(vR_Unit_Conversion) - unknown conversion', convert)
			return 'NaN'

	def ICRS_EQ2ICRScartesian(self):
		"""
		 This function accepts the RA and Dec coordinates as seen from Sun's Centre (in units of degrees) as well as 
		 	the parallax of the object of concern (or heliocentric distance) from ICRS and the radial velocity and
		 	returns the ICRS cartesian coordinates and velocities of the object.

		Input: 
			RA (integer, float, or string, units of deg)	

			Dec (integer, float, or string, units of deg)	

			parallax (integer, float, or string, units of miliarcsec) or helio_dist (integer, float, or string, AU)

			v_R (integer, float, or string, units of AU/yr)


		Outputs: 
			cart_p - array of 3 values ([x,y,z])
				   - Units of length (AU)

				x (float)	

				y (float)

				z (float)

			cart_v - array of 3 values ([vx,vy,vz])
				   - Units of length (depends on v_R)

				vx (float)	

				vy (float)

				vz (float)			

		Example of use:

			cart = 

			returns 

		"""

		if self.helio_dist == []:

			# Checking for type and correcting if necessary
			try:
				RA = float(self.ra)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(RA))
				return 'NaN'
			try:
				Dec = float(self.dec)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(Dec))
				return 'NaN'
			try:
				parallax = float(self.para)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(parallax))
				return 'NaN'
			try:
				v_R = float(self.v_R)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(v_R))
				return 'NaN'
			try:
				pmra = float(self.pmra)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(pmra))
				return 'NaN'
			try:
				pmdec = float(self.pmdec)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(pmdec))
				return 'NaN'

			d = parallax2dist(parallax)
			self.helio_dist = d

		else:

			# Checking for type and correcting if necessary
			try:
				RA = float(self.ra)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(RA))
				return 'NaN'
			try:
				Dec = float(self.dec)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(Dec))
				return 'NaN'
			try:
				d = float(self.helio_dist)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(d))
				return 'NaN'
			try:
				v_R = float(self.v_R)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(v_R))
				return 'NaN'
			try:
				pmra = float(self.pmra)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(pmra))
				return 'NaN'
			try:
				pmdec = float(self.pmdec)
			except ValueError:
				print('WARNING!: TypeError(ICRS2galactocentric_cartesian) - input format not a float', type(pmdec))
				return 'NaN'

			d = self.helio_dist

		import numpy as np

		# position (from astropy)
		x = d*np.cos(np.radians(RA))*np.cos(np.radians(Dec))
		y = d*np.sin(np.radians(RA))*np.cos(np.radians(Dec))
		z = d*np.sin(np.radians(Dec))

		# velocity (from P. A. Dybczynski and F. Berski, 2015)
		# pc2AU - same as arcsec to radians. Since d is in AU, want pmra in radians/yr, so divide by pc2AU
		v1 = d*(pmra*10**-3)/self.uCon.pc2AU /np.cos(np.radians(Dec))
		v2 = d*(pmdec*10**-3)/self.uCon.pc2AU
		v3 = v_R
		vx = -v1*np.sin(np.radians(RA))*np.cos(np.radians(Dec)) - v2*np.sin(np.radians(Dec))*np.cos(np.radians(RA)) + v3*np.cos(np.radians(RA))*np.cos(np.radians(Dec))
		vy = v1*np.cos(np.radians(RA))*np.cos(np.radians(Dec)) - v2*np.sin(np.radians(RA))*np.sin(np.radians(Dec)) + v3*np.sin(np.radians(RA))*np.cos(np.radians(Dec))
		vz = v2*np.cos(np.radians(Dec)) + v3*np.sin(np.radians(Dec))

		self.ICRScart_p = np.array([x,y,z])

		self.ICRScart_v = np.array([vx,vy,vz])

	def ICRS2galactocentric_cartesian_old(self, GC_ra=266.4051, GC_dec=-28.936175, GC_roll=58.5986320306, GC_d=8200, z_sun=25, v_sun=[10, 248, 7]):
		"""
		 This function accepts the RA and Dec coordinates as seen from Sun's Centre (in units of degrees) as well as 
		 	the parallax of the object of concern from ICRS and the radial velocity, and returns the galactocentric cartesian coordiantes and
		 	and velocities of the object.

		Input: 
			RA (integer, float, or string, units of deg)	

			Dec (integer, float, or string, units of deg)	

			parallax (integer, float, or string, units of arcsec)

			v_R (integer, float, or string, units of km/s)

			or

			ICRS cartesian vectors

		Optional:

			GC_ra   - Change value of GC_ra (deg) [default: 266.4051]
			GC_dec  - Change value of GC_dec (deg) [default: -28.936175]
			GC_roll - Change value of GC_roll (deg) [default: 58.5986320306]
			GC_d    - Change value of GC_d (pc) [default: 8200]
			z_sun   - Change value of z_sun (pc) [default: 25]
			v_sun   - Change value of v_sun (km/s) [default: [ 10, 248, 7 ] ]

			(last 3 from Bland-Hawthorn & Gerhard, "The Galaxy in Context: 
			Structural, Kinematic and Integrated Properties", 2016)
			
			Astropy:
			GC_roll = 58.5986320306 deg
			GC_d = 8122 pc
			z_sun = 20.8 pc
			v_sun = [12.9, 245.6, 7.78]


		Outputs: 
			galcart_p - array of 3 values ([x,y,z])
			  	      - Units of length (AU)

				x (float)	

				y (float)

				z (float)

			galcart_v - array of 3 values ([vx,vy,vz])
				      - Units of length (depends on v_R)

				vx (float)	

				vy (float)

				vz (float)			

		Example of use:

			cart = 

			returns 

		"""

		import numpy as np

		if len(self.ICRScart_v) == 0 and len(self.ICRScart_p) == 0:

				self.ICRS_EQ2ICRScartesian()

		else:

			# Checking for type and correcting if necessary
			if type(self.ICRScart_p) != type(np.array([1,1,1])) or len(self.ICRScart_p) != 3:
				print('WARNING!: ValueError(ICRS2galactocentric_cartesian) - input format not an array of length 3. Type:', type(self.ICRScart_p), ' length: ', len(self.ICRScart_p))
				return 'NaN'
			if type(self.ICRScart_v) != type(np.array([1,1,1])) or len(self.ICRScart_v) != 3:
				print('WARNING!: ValueError(ICRS2galactocentric_cartesian) - input format not an array of length 3. Type:', type(self.ICRScart_v), ' length: ', len(self.ICRScart_v))
				return 'NaN'

		# Converting default values from km/s to AU/yr and pc to AU
		v_sun = np.array(v_sun)*self.uCon.kms2AUyr
		GC_d = GC_d*self.uCon.pc2AU
		z_sun = z_sun*self.uCon.pc2AU

		ICRScart_p = self.ICRScart_p
		ICRScart_v = self.ICRScart_v

		# Defining rotation matrices:
		# From roll
		R3 = np.array([[1,0,0], [0, np.cos(np.radians(GC_roll)), np.sin(np.radians(GC_roll))],
				[0, -np.sin(np.radians(GC_roll)), np.cos(np.radians(GC_roll))]])

		# from galactic centre coordinates
		R1 = np.array([[np.cos(np.radians(GC_dec)),0,np.sin(np.radians(GC_dec))],[0,1,0],
				 [-np.sin(np.radians(GC_dec)), 0, np.cos(np.radians(GC_dec))] ])

		R2 = np.array([[np.cos(np.radians(GC_ra)),np.sin(np.radians(GC_ra)),0],
				 [-np.sin(np.radians(GC_ra)), np.cos(np.radians(GC_ra)),0], [0,0,1] ])

		R = np.matmul(R3,np.matmul(R1,R2))

		# Distance to galactic centre, along x-axis:
		d_GC_vec = np.array([GC_d,0,0])

		# intermediate step
		int_r = np.matmul(R,ICRScart_p) - d_GC_vec

		int_v = np.matmul(R,ICRScart_v) + v_sun

		# Rotation angle about new y axis
		theta = np.arcsin(z_sun/GC_d)

		H = np.array( [[np.cos(theta),0,np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]] )

		# Final galactorcentric coordiantes
		galcart_p = np.matmul(H,int_r)

		galcart_v = np.matmul(H,int_v)
		
		self.galcart_p = galcart_p
		self.galcart_v = galcart_v

	def ICRS2galactocentric_cartesian(self, GP_ra=192.8594812, GP_dec=27.12852897, GP_theta = 122.93191857, GC_d=8200, z_sun=25, v_sun=[10, 248, 7]):
		"""
		 This function accepts the RA and Dec coordinates as seen from Sun's Centre (in units of degrees) as well as 
		 	the parallax of the object of concern from ICRS and the radial velocity, and returns the galactocentric cartesian coordiantes and
		 	and velocities of the object.

		Input: 
			RA (integer, float, or string, units of deg)	

			Dec (integer, float, or string, units of deg)	

			parallax (integer, float, or string, units of arcsec)

			v_R (integer, float, or string, units of km/s)

			or

			ICRS cartesian vectors

		Optional:

			GC_ra   - Change value of GC_ra (deg) [default: 266.4051]
			GC_dec  - Change value of GC_dec (deg) [default: -28.936175]
			GC_roll - Change value of GC_roll (deg) [default: 58.5986320306]
			GC_d    - Change value of GC_d (pc) [default: 8200]
			z_sun   - Change value of z_sun (pc) [default: 25]
			v_sun   - Change value of v_sun (km/s) [default: [ 10, 248, 7 ] ]

			(last 3 from Bland-Hawthorn & Gerhard, "The Galaxy in Context: 
			Structural, Kinematic and Integrated Properties", 2016)
			
			Astropy:
			GC_roll = 58.5986320306 deg
			GC_d = 8122 pc
			z_sun = 20.8 pc
			v_sun = [12.9, 245.6, 7.78]


		Outputs: 
			galcart_p - array of 3 values ([x,y,z])
			  	      - Units of length (AU)

				x (float)	

				y (float)

				z (float)

			galcart_v - array of 3 values ([vx,vy,vz])
				      - Units of length (depends on v_R)

				vx (float)	

				vy (float)

				vz (float)			

		Example of use:

			cart = 

			returns 

		"""

		import numpy as np

		if len(self.ICRScart_v) == 0 and len(self.ICRScart_p) == 0:

				self.ICRS_EQ2ICRScartesian()

		else:

			# Checking for type and correcting if necessary
			if type(self.ICRScart_p) != type(np.array([1,1,1])) or len(self.ICRScart_p) != 3:
				print('WARNING!: ValueError(ICRS2galactocentric_cartesian) - input format not an array of length 3. Type:', type(self.ICRScart_p), ' length: ', len(self.ICRScart_p))
				return 'NaN'
			if type(self.ICRScart_v) != type(np.array([1,1,1])) or len(self.ICRScart_v) != 3:
				print('WARNING!: ValueError(ICRS2galactocentric_cartesian) - input format not an array of length 3. Type:', type(self.ICRScart_v), ' length: ', len(self.ICRScart_v))
				return 'NaN'

		# Converting default values from km/s to AU/yr and pc to AU
		v_sun = np.array(v_sun)*self.uCon.km2AU/self.uCon.s2yr
		GC_d = GC_d*self.uCon.pc2AU
		z_sun = z_sun*self.uCon.pc2AU

		ICRScart_p = self.ICRScart_p
		ICRScart_v = self.ICRScart_v

		# Distance to galactic centre, along x-axis:
		d_GC_vec = np.array([-GC_d,0,z_sun])

		# defining rotation matrix
		M1 = rot('z',np.radians(90+GP_ra))
		M2 = rot('x',np.radians(90-GP_dec))
		M3 = rot('z',np.radians(90-GP_theta))

		M = np.matmul(np.matmul(M3,M2),M1)
		
		r = np.matmul(M,ICRScart_p) + d_GC_vec
		v = np.matmul(M,ICRScart_v) + v_sun

		self.galcart_p = r
		self.galcart_v = v



def initiateSun(SunGC_param, SolarEjectFlag):
	"""
	This function takes the Sun's galactocentric parameters and ejection flag and returns a Sun object.

	Input:
		SunGC_param - array of 3 values ([GC_d, z_sun, v_sun])
					  - Units of length (AU)
					  - Units of velocity (km/s) (v_sun is vector)
	"""
	Sun = galacticCoord('Sun')
	Sun.epoch = 2015.5
	Sun.ra = 0
	Sun.dec = 0
	Sun.helio_dist = 0
	Sun.v_R = 0
	Sun.pmra = 0
	Sun.pmdec = 0
	Sun.mass = 1
	Sun.CloseApproachFlag = 1
	Sun.vR_Unit_Conversion('kms2AUyr')
	Sun.ICRS2galactocentric_cartesian(GC_d=SunGC_param[0], z_sun=SunGC_param[1], 
		v_sun=SunGC_param[2])
	Sun.ejectFlag=SolarEjectFlag

	return Sun



# TESTING!!
if __name__ == '__main__':
	import numpy as np

	star1 = galacticCoord('Proxima Cen')

	star1.epoch = 2016.0
	star1.ra = 217.392321472009
	star1.dec = -62.6760751167667
	star1.para = 768.066539187357
	star1.v_R = -22.345
	star1.pmra = -3781.74100826516
	star1.pmdec = 769.465014647862
	star1.vR_Unit_Conversion('kms2AUyr')

	star1.ICRS_EQ2ICRScartesian()
	print(star1.ICRScart_p,'\n',star1.ICRScart_v)

	star1.ICRS2galactocentric_cartesian()
	print(star1.galcart_p,'\n',star1.galcart_v)


	Sun = galacticCoord('Sun')
	Sun.epoch = 2016.0
	Sun.ra = 0
	Sun.dec = 0
	Sun.helio_dist = 0
	Sun.v_R = 0
	Sun.pmra = 0
	Sun.pmdec = 0
	Sun.vR_Unit_Conversion('kms2AUyr')
	# Sun.ICRS_EQ2ICRScartesian()
	# print(Sun.ICRScart_p,'\n',Sun.ICRScart_v)

	Sun.ICRS2galactocentric_cartesian()
	print(Sun.galcart_p,'\n',Sun.galcart_v)




	# GC= galacticCoord('GC')
	# GC.epoch = 0
	# GC.ra = 266.4051
	# GC.dec = -28.936175
	# GC.helio_dist = 8122*206264.8075
	# GC.v_R = 46
	# GC.pmra = -2.7
	# GC.pmdec = -5.6
	# GC.vR_Unit_Conversion('kms2AUyr')
	# GC.ICRS_EQ2ICRScartesian()
	# print(GC.ICRScart_p,'\n',GC.ICRScart_v)
	# GC.ICRS2galactocentric_cartesian()
	# print(GC.galcart_p,'\n',GC.galcart_v)



	print('===============================')
	star2= galacticCoord('star2')
	star2.epoch = 0
	star2.ra = 89.014303
	star2.dec = 13.924912
	star2.para = 37.59
	star2.v_R = 0.37
	star2.pmra = 372.72/np.cos(np.radians(13.924912))
	star2.pmdec=-483.69
	star2.vR_Unit_Conversion('kms2AUyr')
	# star2.ICRS_EQ2ICRScartesian()
	# print(star2.ICRScart_p,'\n',star2.ICRScart_v)
	star2.ICRS2galactocentric_cartesian()
	print(star2.galcart_p,'\n',star2.galcart_v)


