import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from src.settings import save_as_pickle, load_pickle
import glob
import numpy as np
from sympy import DiracDelta
from scipy.interpolate import interp1d
from matplotlib.patches import Patch


# Contains no data directly relevant to thesis. Images for presentation are plotted here

# Img 1

# points = np.array([(-9,3),(-8,3),(-7,3),(-6,3),(-5,3),(-4,3),(-3,3),(-2,3),(-1,2),(0,0),(1,2),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3)])
#
# x = points[:,0]
# y = points[:,1]
#
# z = np.polyfit(x, y, 10)
# f = np.poly1d(z)
x = np.linspace(-10 , 10, 50)
f = - DiracDelta(x)
plt.plot(x, f, '-')
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['interpolated'], loc='best')
plt.show()
print("fin")