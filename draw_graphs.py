"""Barebone script to draw pareto graphs"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline,splrep,splev,interp1d




iteration = np.array([10,15,20,25,30,35,40,45,50,75,100])
loss = np.array([0.297441558, 0.359849212, 0.319080556, 0.184535086, 0.289120287
,0.202361349, 0.233063436, 0.159715685
, 0.183388828, 0.208292567, 0.303400268])

xnew = np.linspace(iteration.min(), iteration.max(), 300)
spl = splrep(iteration,loss,k=2)
#spl = make_interp_spline(iteration, loss, k=2)  # type: BSpline
#loss_smooth = spl(xnew)
#f2 = interp1d(iteration, loss, kind='cubic')
fit2 = np.polyfit(iteration, loss,3)
f2 = np.poly1d(fit2)
#loss_smooth = splev(xnew, spl)
loss_smooth = f2(xnew)


plt.plot(iteration,loss, 'x')
plt.plot(xnew,loss_smooth)
plt.show()















