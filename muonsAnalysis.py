import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

low = 0
high = 2048
cm_20 = open("20 cm_30mins_031419_2.SPE", 'r')
cm_60 = open("60cm_30mins_031419_2.SPE", 'r')
cm_100 = open("100cm_60mins_031419_3.SPE", 'r')
cm_140 = open("140cm_45min_031519.SPE", 'r')
cm_180 = open("180cm_45mins_031519.SPE", 'r')
cm_222 = open("222cm_115_031519.SPE", 'r')
cm_260 = open("260cm_45min_031519.SPE", 'r')
cm_340 = open("340cm_1025_031519.SPE", 'r')




heights = [20,60,100,140,180,222,260,340]
height_data = [i.readlines() for i in [cm_20, cm_60, cm_100, cm_140, cm_180, cm_222, cm_260, cm_340]]
updated_height = []

for each in height_data:
    next = []
    for i in range(low, high):
        next.append(int(each[i+12].rstrip('\n').lstrip(' ')))
    updated_height.append(next)
raw_data = []
errors = []
# print(updated_height[0])
for j in range(8):
    raw_data = []
    for i in range(len(updated_height[j])):
        raw_data.extend([i for _ in range(updated_height[j][i])])
    errors.append(np.std(raw_data)/np.sqrt(len(raw_data)))
# print(raw_data)
# print(errors)

fig = plt.figure()
colors = ['red','blue', 'green', 'black', 'yellow', 'purple', 'orange', 'brown']
axe =fig.add_subplot(111)




means = [sum([updated_height[j][int(i)]*i for i in np.linspace(low,high-1,high)])/sum(updated_height[j]) for j in range(len(updated_height))]
for i in range(len(updated_height)):
        axe.plot(np.linspace(low,high+1,high-low), [j*100 for j in updated_height[i]], linewidth=1, color=colors[i])
        axe.scatter(means[i], 140, color = colors[i], s=1)

avg_slant = [24,61.7,100.4, 140.1, 180, 222 , 260, 340]

speed = [avg_slant[i]/(means[i]/20/(10**9)) for i in range(len(means))]
# print(3.2*(10**9)/(123.4/20)/299780000)
speeds = []
for i in range(len(avg_slant)):
    for j in range(len(avg_slant)):
        if i != j and j >i:
            speeds.append((i,j,((heights[j]-heights[i])/100*(10**9)*20/(means[j]-means[i])/299780000)))

# print(speeds)
# print([mean*3/80 for mean in means])
# print(np.average(speeds))

def linear(x, m, b):
    return m*x + b


fig = plt.figure()
axe = fig.add_subplot(111)
axe.scatter(avg_slant, [mean/20 for mean in means])
# plt.show()
popt,pcov = curve_fit(linear, avg_slant[1:-1],[mean/20 for mean in means][1:-1], maxfev=20000, sigma=[i/20 for i in errors[1:-1]] )
axe.plot(np.linspace(0, 350, 10000), [linear(x,0.0337 , 29.4) for x in np.linspace(0,350, 10000)])
axe.plot(np.linspace(0, 350, 10000), [linear(x,popt[0] , popt[1]) for x in np.linspace(0,350, 10000)], color='red')
for i in range(len(avg_slant)):
    axe.errorbar(avg_slant[i], means[i]/20, xerr=0, yerr=errors[i]/20)
# plt.show()
# print(np.sqrt(pcov[0][0]))
# print(np.sqrt(pcov[0]))


muons = open('decay_3.SPE', 'r')
info = muons.readlines()
data = []
for i in range(2048):
    data.append(int(info[i+12].rstrip('\n').lstrip(' ')))
# print(data)
x = np.linspace(0,2047,2048)
actual_data = []
for val in x:
    if data[int(val)] != 0:
        actual_data.extend([int(val) for _ in range(data[int(val)])])
raw = []
# for i in range(len(actual_x)):
#     raw.extend([i for _ in range(actual_x[i])])
# print(raw)
fig = plt.figure()
axe = fig.add_subplot(111)
y_hist, x_hist, patches = axe.hist(actual_data, bins=35, edgecolor='black')
def func(x, t, a, b, c):
    return b*np.exp(-(x-c)/t)+a
y_hist = list(y_hist)[4:]
x_hist = list(x_hist)[4:-1]
x_hist = [i + 25 for i in x_hist]
# print(len(y_hist))
# print(len(x_hist))
axe.scatter(x_hist,y_hist)

popt, pcov = curve_fit(func, xdata = x_hist, ydata = y_hist, sigma = [np.sqrt(i) for i in y_hist], maxfev=20000, p0 = [500,0, 700, 220])
axe.plot(x, func(x, popt[0],popt[1], popt[2],popt[3]))

print(popt)


def get_chi(func, xdata, ydata, error):
    chis = []
    chi=0
    for x,y, e in zip(xdata, ydata, error):
        val = func(x)
        new = ((val-y)/e)**2
        chi += new
        chis.append(new)
    return chi, chis
axe.set_ylim(0,750)
print(get_chi(lambda x: func(x, popt[0], popt[1], popt[2], popt[3]), x_hist, y_hist,error = [np.sqrt(i) for i in y_hist])[0]/27)
plt.show()





