# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy.stats import poisson

# returns chi2 value
def chisqr(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
    return chisqr



loc = r"C:/Users/erikp/Desktop/School/Classes/Course 8/8.13/Poisson/Poisson_Data.csv"

data = pd.read_csv(loc, na_filter=True)


# individual tests and long exposure runs
tests  = [data.loc[:,'Test 1'].values, data.loc[:,'Test 3'].values,data.loc[:,'Test 5'].values,data.loc[:,'Test 7'].values]
longs = [data.loc[:0,'Test 2'].values[0], data.loc[:0,'Test 4'].values[0],data.loc[:0,'Test 6'].values[0],data.loc[:0,'Test 8'].values[0]]


# cumulative means of each test
cumulative1 = [np.mean(tests[0][:i+1]) for i in range(len(tests[0]))]
cumulative2 = [np.mean(tests[1][:i+1]) for i in range(len(tests[1]))]
cumulative3 = [np.mean(tests[2][:i+1]) for i in range(len(tests[2]))]
cumulative4 = [np.mean(tests[3][:i+1]) for i in range(len(tests[3]))]
cums = [cumulative1, cumulative2, cumulative3, cumulative4]


fig = plt.figure()

# gets means and std for each data set
means = [np.mean(i) for i in tests]
stdvs = [np.std(i, ddof=1) for i in tests]

# estimated via long run
mu = 146
x = np.linspace(100, 200, 1000)
y = [10*len(tests[3])/np.sqrt(2*math.pi*mu)*np.e**(-(i-mu)**2/(2*mu)) for i in x]

ax = fig.add_subplot(111)

ax.plot(x, y)

# bins data and plots in a histogram
heights, widths, patches = ax.hist(tests[3], bins=range(100, 200, 10), normed=False, linewidth=2, edgecolor = 'black')
for i in range(len(heights)):
    ax.errorbar(widths[i]+5, heights[i], xerr=0, yerr=np.sqrt(heights[i]),fmt='-o', ecolor='black')

for i in range(len(heights)):
    deviation = (heights[i]-10*len(tests[3])/np.sqrt(2*math.pi*mu)*np.e**(-(widths[i]+5-mu)**2/(2*mu)))**2
    er = heights[i]
    if er != 0:
        print(deviation/er)
        chi += deviation/er

plt.show()

