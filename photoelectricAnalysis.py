# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy as sp
from scipy.optimize import curve_fit
import math
import pandas as pd

#file paths for each wavelength
loc_365 = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Photoelectric Effect/photo_data_365.csv"
loc_404 = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Photoelectric Effect/photo_data_404.csv"
loc_435 = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Photoelectric Effect/photo_data_435.csv"
loc_546 = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Photoelectric Effect/photo_data_546.csv"
loc_577 = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Photoelectric Effect/photo_data_577.csv"

# all the file paths to the individual sheets
sheet_locs = [loc_365, loc_404, loc_435, loc_546, loc_577]

full_data = [pd.read_csv(sheet, header=None) for sheet in sheet_locs]
listed_data = [[list(sheet.loc[:,i]) for i in range(1, len(sheet.columns))] for sheet in full_data]

"""
listed_data is now in the form: it is a list of lists. each outer list
is one of the wavelength, i.e., 
    listed_data[0] is wavelength 365
    listed_data[1] is wavelength 404
    listed_data[2] is wavelength 435
    ...... etc
    
each list is full full of lists of the form (voltage, measure1, measure2, measure3, ....)
"""

plotting_data = []  # data in the proper form to plot
for wavelength in listed_data:
    y = []
    x = []
    for data in wavelength:
        x.append(data[0])
        y.append(np.average(data[1:]))
    plotting_data.append((x.copy(), y.copy()))


labels = ['$\lambda = 365 nm$','$\lambda = 404 nm$', \
         '$\lambda = 435 nm$', '$\lambda = 546 nm$', \
          '$\lambda = 577 nm$']  # wavelengths we were measuring at

fig = plt.figure()

for i in range(len(plotting_data)):
    axe = fig.add_subplot(2,3,i+1)  # creating one plot for each of the wavelengths
    axe.plot(plotting_data[i][0], plotting_data[i][1], label= labels[i])
    axe.scatter(plotting_data[i][0], plotting_data[i][1], s=10)


#plot formatting
plt.grid()
plt.legend(loc='upper right', title='wavelength:')
plt.title("Applied Voltage vs. Photocurrent")
plt.xlabel('Applied Voltage (V)')
plt.ylabel('Measured Current (nA)')
plt.show()

slopes = []
for each in plotting_data:
    slopes.append((each[0][1]-each[0][0])/(each[1][1]-each[1][0]))

waves = [365*(10**(-9)), 404.7*(10**(-9)), 435.8*(10**(-9)), 546.1*(10**(-9)), 577*(10**(-9))]
wave_error = [0.2*(10**(-9))]*5

stopping_volt = [1.1, 0.8, 0.75, 0.65, 0.55]
stopping_error = [0.2]*5

frequency = [3*(10**8)/wave for wave in waves]
frequency_error = [3*(10**8)/(waves[i]**2)*wave_error[i] for i in range(len(waves))]

# performing a linear fit of the data
def linear(x, slope, yint):
    return x*slope + yint

#optimizing our linear fit with the error found
popt, pcov = curve_fit(linear, xdata = frequency, ydata = stopping_volt, p0=[4*(10**(-15)), -2], sigma=stopping_error, maxfev=20000 )

