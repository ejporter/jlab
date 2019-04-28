import numpy as np
import matplotlib.pyplot as plt

file_obj = open("Cobalt_60_spectrum.SPE", 'r')
x = np.linspace(0, 2046, 2047)
y= []
lines = file_obj.readlines()
for i in range(2047):
    print(i)
    y.append(int(lines[i+13].rstrip('\n').lstrip(' ')))
fig = plt.figure()
axe =fig.add_subplot(111)
axe.plot(x, y, linewidth=1)
plt.show()
