# Python program to Plot Circle

# importing libraries
import numpy as np
from matplotlib import pyplot as plt

# Creating equally spaced 100 data in range 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Setting radius
radius = 5

# Generating x and y data
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Plotting
plt.plot(x, y)
plt.axis('equal')
plt.title('Circle')
plt.show()


import math
pi = math.pi

def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]


