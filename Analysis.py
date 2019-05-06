import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit




#all file names. This is in a dictionary, so access a certain subsection of files by using files['calibration'], files['sodium'], etc.
'''
taken out

		'calibration_mercury_firstDoublet_53_1'

'calibration_neon_triple_2_53'
'''


files = {'calibration': ['calibration_neon_triple_1_53',\
		
		'calibration_neon_triple_3_53', \
        'calibration_mercury_3560_triple_53_1',\
		'calibration_mercury_3560_triple_53_2',\
        'calibration_mercury_3560_triple_53_3', \
        'calibration_mercury_4046_53',\
		'calibration_mercury_4358_53',\
        'calibration_mercury_5460_53', \
        'calibration_mercury_yellowDoublet_53_1',\
		'calibration_mercury_yellowDoublet_53_2'],\
        'Deut':['DeutHydro_32_53_1',\
		'DeutHydro_32_53_2',\
		'DeutHydro_42_53_1',\
		'DeutHydro_42_53_2',\
        'DeutHydro_52_53_1',\
		'DeutHydro_52_53_2',\
		'DeutHydro_62_53_1',\
		'DeutHydro_62_53_2'],\
        'sodium': ['SodiumDoublet_DLines_53_2',\
		'SodiumDoublet_DLines_53_1',\
		'SodiumDoublet_Green_53_1', \
        'SodiumDoublet_Green_53_2',\
		'SodiumDoublet_Red_53_1', \
		'SodiumDoublet_Red_53_2', \
		'SodiumDoublet_4978_53_2', \
		'SodiumDoublet_4978_53_1']}
		
		
		





def plotFile(fileName):
	"""
	takes in the name of a file in the same folder as the script and plots it
	"""
	# sets up the plotting
	fig = plt.figure()
	axe = fig.add_subplot(111)
	
	with open(fileName, 'r') as data:
	
		raw = []
		for each in data.readlines():
			elts = each.split('\t')
			raw.append((float(elts[1]), float(elts[2].rstrip('\n'))))
		#gets the raw data in the form [(wave, height), (wave, height),...]
			
		plt.title(fileName)
		axe.plot([x[0] for x in raw], [y[1] for y in raw])  #plots wavelength versus counts
		
		plt.show()






def getFit(fileName):
	"""
	returns the parameters for a gaussian fit in the form [amp, mu, std, off]
	"""

	def gaussian(x, amp, mu, std, off):
		return amp/np.sqrt(2*np.pi*std**2)*np.e**(-(x-mu)**2/2/std**2)+off  # function to be fit

	with open(fileName, 'r') as data:
		
		#gets the data into the lists xdata and ydata
		raw = []
		for each in data.readlines():
			elts = each.split('\t')
			raw.append((float(elts[1]), float(elts[2].rstrip('\n'))))

		xdata = [x[0] for x in raw]
		ydata = [x[1] for x in raw]
		
		
		popt, pcov = curve_fit(gaussian, xdata, ydata, p0 = [int(max(ydata)), int(xdata[ydata.index(max(ydata))])-0.2, 1, 0], maxfev=40000)
		return popt
		#returns the parameter for the gaussian fit
		
def plotFit(fileName):

	"""
	plots the file along with its gaussian fit
	"""

	def gaussian(x, amp, mu, std, off):
		return amp/np.sqrt(2*np.pi*std**2)*np.e**(-(x-mu)**2/2/std**2)+off
	
	params = getFit(fileName)  # gets the paramaters of the fit

	fig = plt.figure()
	axe = fig.add_subplot(111)
	
	
	with open(fileName, 'r') as data:
	
		# gets the data in the right form
		raw = []
		for each in data.readlines():
			elts = each.split('\t')
			raw.append((float(elts[1]), float(elts[2].rstrip('\n'))))
		xdata = [x[0] for x in raw]
		ydata = [x[1] for x in raw]
		
		# plots the raw data
		axe.plot(xdata, ydata)
		
		# plots the gaussian fit along the range of the raw data
		xd=  np.linspace(xdata[0], xdata[-1], 1000)
		axe.plot(xd, [gaussian(x, params[0], params[1], params[2], params[3]) for x in xd], color='red')
		plt.title(fileName)
		plt.show()

	
# 12 colors you can use to color code the calibration if you want	
colors = ['orange', 'red', 'green', 'blue', 'black', 'm', 'cyan', 'brown', 'purple', 'silver', 'yellow', 'pink']



def plotCalibration(calibs):
	"""
	does a fit of all the calibration files, returns the best fit function
	"""
	def getChi(func, xdata, ydata, error):
	# returns the chi2 values and total for the plot
		chis = []
		for i in range(len(xdata)):
			err = func(xdata[i])-ydata[i]
			chis.append((err/error[i])**2)
		return chis, sum(chis)

	
	# sets up the plot
	fig = plt.figure()
	axe = fig.add_subplot(211)
	
	# pulls out the mean of each calibration files
	means = []
	error = []
	for each in calibs:
		params = getFit(each)
		means.append(params[1])
		error.append(params[2])
		
	# weird one I dont want to worry about fixing
	means[1] = 6258.01
	error[1] = 0.06

	#actual values for all of the calibration wavelengths
	actual = [6266.495, 6304.7889, 3650.153, 3654.836, 3663.279, 4046.563, 4358.328, 5460.735, 5769.598, 5790.663]
	

	'''
	taken out
, 3131.548
, 6328.1646
	'''

	dif = [actual[i] - means[i] for i in range(len(means))]
	# plots the calibration points
	axe.scatter(means,dif , color='red', marker='x')
	
	
	
	# defining functions for the fits
	def lin(x, m, b):
		return m*x+b
		
	def quad(x, a, b, c):
		return a*x**2+b*x+c
		
	def cub(x, a, b, c, d):
		return a*x**3+b*x**2+c*x+d


	# fits the parameters
	popt, pcov = curve_fit(quad, means, dif, maxfev=20000, sigma=error)
		
		
	# plots the fit
	x = np.linspace(3000,6500, 10000)
	axe.plot(x, [quad(i, popt[0], popt[1], popt[2]) for i in x], color='blue')
	
	
	# sets up the residual plot
	axe2 = fig.add_subplot(212)
	axe2.plot(x, [0]*len(x))
		
	#axe2.scatter(means, [dif[i]-quad(means[i], popt[0], popt[1], popt[2]) for i in range(len(means))], color = 'black', marker = 'v')
	for i in range(len(means)):
		axe2.errorbar(means[i], dif[i]-quad(means[i], popt[0], popt[1], popt[2]), xerr = 0, yerr = error[i], color = 'purple', fmt = '--o', capsize = 5)
	
	print('A = ' + str(popt[0]))
	print('B = ' + str(popt[1]))
	print('C = ' + str(popt[2]))
	
	
	func = lambda x: quad(x,popt[0], popt[1], popt[2])

	chi = getChi(func, means, dif, error)
	print('Chi2 = ' + str(chi[1]/7))
	#returns the functional fit
	return func






fit = plotCalibration(files['calibration'])

rydFiles = [files['Deut'][2*i+1] for i in range(len(files['Deut'])//2)]

means = [getFit(i)[1]+fit(getFit(i)[1]) for i in rydFiles]
jumps = [3,4,5,6]
ryd = []
for i in range(len(means)):
	levels = (1/4-1/jumps[i]**2)
	ryd.append(1/means[i]/levels)
act = 10967759.3
print('Rydberg = ' + str(np.average(ryd)))
print('Error (percent) = ' + str(100*(np.average(ryd)*10**10-act)/act))

plt.show()



"""
Now were getting the mass ratio
"""

nf = 2 # balmer series
ni = [3,4,5,6]
mh = 1.6726219*10**(-27)
me = 9.10938356*10**(-31)
shifts = []
print('\n\n\n')
for i in range(len(files['Deut'])//2):
	deut = getFit(files['Deut'][2*i])[1]
	deut += fit(deut)
	hydr = getFit(files['Deut'][2*i+1])[1]
	hydr += fit(hydr)
	shift = (hydr-deut)*10**(-10) #angstroms
	shifts.append(shift)


print(np.average(ryd)*10**10)
ratio = []

shift = 0.176*10**(-9)
rinf = 10973731
print(shift*rinf*mh/me*(0.25-1/9.0))
for i in range(4):
	ratio.append(1- shifts[i]*np.average(ryd)*10**10*(mh+me)/me*(1/4.0-1/float(ni[i]**2)))

print(ratio)
print("Ratio = " +str(np.average(ratio)))
print((np.average(ratio)-0.4963)/0.4963)
