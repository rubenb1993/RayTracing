```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import Axes3D
>>> from scipy.optimize import differential_evolution
>>> import sympy
...
>>> %matplotlib notebook
```

```python
>>> def pol2cart(rho, phi):
...     '''converts polar to cartesian coordinates'''
...
...     x = rho * np.cos(phi)
...     y = rho * np.sin(phi)
...
...     return[x, y]
...
>>> # hier doen we niks mee
... def paraxialFocus(c, n, t):
...     '''Determine the focus in the paraxial limit'''
...     f = sympy.Symbol('f')
...     t = sympy.Matrix(t)
...     t[-1] = f
...     product2 = np.identity(2)
...
...     for i in range(len(t)):
...         T = np.array([[1, t[i]], [0, 1]])
...         product1 = np.dot(T, product2)
...         if i == len(t)-1:
...             break
...
...         R = np.array([[1, 0], [c[i]*(n[i] - n[i+1])/n[i+1], n[i]/n[i+1]]])
...         product2 = np.dot(R, product1)
...     return sympy.solve(product1[0, 0], f)[0]
...
...
>>> def incomingBeam(Rmax, Nr, Ntheta, angle):
...     # Circular starting pattern (could also be set to random if needed)
...
>>> #     r = np.array([np.linspace(0.5, Rmax, Nr) for i in range(Ntheta)])
... #     theta = np.array([np.linspace(0, 2*np.pi, Ntheta) for i in range(Nr)]).T
... #     [X, Y] = pol2cart(r, theta)
...     r = np.array([])
...     theta = np.array([])
...     for i in range(Nr):
...         R = (i+1)*Rmax/Nr
...         j = (i+1)*5
...         a = np.array([R]*j)
...         r = np.append(r,a)
...         b = np.linspace(0, 2*np.pi, j)
...         theta = np.append(theta, b)
...
...     [X, Y] = pol2cart(r, theta)
...
...     X = X.reshape(-1)
...     Y = Y.reshape(-1)
...
...     # Define unit vector in z-dimension
...     k = np.array([[0, 0, 1]])
...
...     # Position and angle vector (place, coordinate, beam)
...     d = np.zeros(shape = (Nsteps+1, 3, len(X)))
...     d[0, 0] = X
...     d[0, 1] = Y
...
...     u = np.zeros(shape = (Nsteps+1, 3, len(X)))
...     u[0, :] = np.array([[angle], [0], [0]])
...     u[0, 2] = np.sqrt(1 - u[0, 0]**2 - u[0, 1]**2) # Calculate angle with z-axis based on x and y axes
...
...     return d, u, k
...
...
>>> def traceRays(c, t, n, d, u, k):
...     '''Compute the ray paths'''
...
...     for i in range(Nsteps):
...         # Algorithm to obtain new positions
...         e = t[i]*np.dot(k, u[i]) - np.sum(d[0]*u[0], axis = 0) # A scalar for obtaining the total travel distance T
...         M = d[i] + e*u[i] - t[i]*k.T
...         M2 = np.sum(M*M, axis = 0)
...         cos_inc_angle = np.sqrt(u[i, 2]**2 - c[i]*(c[i]*M2 - 2*M[2]))
...         T = e + (c[i]*M2 - 2*M[2])/(u[i, 2] + cos_inc_angle) # Total path length to interface
...         # Translate d over distance T. Remove z-component to be in the plane perpendicular to z at point z=ti
...         d[i+1] = d[i] + T*u[i] - t[i]*k.T
...
...         # At the last plane, don't calculate the new angles
...         if i == Nsteps-1:
...             break
...
...         # Algorithm to calculate new angle due to refraction
...         mu = n[i]/n[i+1]
...         cos_out_angle = np.sqrt(1 - mu**2 * (1 - cos_inc_angle**2))
...         g = cos_out_angle - mu*cos_inc_angle
...         u[i+1] = mu*u[i] - c[i]*g*d[i+1] + g*k.T
...
...     return d
...
...
>>> def fitness(d):
...     '''Calculate the fitness of the spot in the image plane'''
...     sumDistance = 0
...     N = len(d[0, 0, :]) # Number of rays
...
...     for i in range(N):
...         for j in range(i+1, N):
...             sumDistance += (d[-1, 1, i] - d[-1, 1, j])**2 + (d[-1, 0, i] - d[-1, 0, j])**2
...
...     fitness = np.sqrt(2/(3*N*(3*N-1)) * sumDistance)
...     return fitness
...
>>> def fitness_vector(d):
...     '''Calculate the fitness of the spot in the image plane'''
...     N = len(d[0, 0, :]) # Number of rays
...     ind = np.triu_indices(N, k=1) #create array of upper triangular matrix indices
...
...     # create distance vector for i > j
...     dist = (d[-1, 0, ind[0]] - d[-1, 0, ind[1]])**2 + (d[-1, 1, ind[0]] - d[-1, 1, ind[1]])**2
...     sumDistance = np.sum(dist)
...     fitness = np.sqrt(2/(3*N*(3*N-1)) * sumDistance)
...
...     return fitness
...
>>> def costFunction(x, plotyn=False):
...     c = x[:Nsteps]
...     t = x[Nsteps:]
...     angles = np.array([0.00300, 0.00225, 0.00150, 0.00075, 0])
...     weights = [0.5, 0.125, 0.125, 0.125, 0.125]
...
...     fitnessFactor = 0
...     spot = np.zeros((len(n), len(angles), 3, 275))
...
...     for i in range(len(n)):
...         for j in range(len(angles)):
...             d, u, k = incomingBeam(7, 10, 25, angles[j])
...             d = traceRays(c, t, n[i], d, u, k)
...             spot[i, j] = d[-1]
...             fitnessFactor += weights[j] * fitness_vector(d)
...
...     if plotyn == True:
...         plotSpot(spot, angles)
...         plotRays(t, d, k)
...
...     return fitnessFactor
...
>>> def plotSpot(spot, angles):
...
...     fig, axarr = plt.subplots(len(angles), len(n)+1, figsize=(12, 15))
...     wavelengths = [0.486, 0.546, 0.656]
...
...     for i in range(len(spot)):
...         axarr[0, i].set_title(r'$\lambda$={}'.format(wavelengths[i]), size=16)
...         for j in range(len(spot[0, :])):
...             axarr[j, i].scatter(spot[i, j, 0], spot[i, j, 1])
...
...             axarr[j, -1].set_title(r'$\theta_i$={0:3f}'.format(angles[j]), size=16)
...             axarr[j, -1].axis('off')
...
...             dx = [spot[i, j, 0].min(), spot[i, j, 0].max()]
...             dy = [spot[i, j, 1].min(), spot[i, j, 1].max()]
...             axarr[j, i].set_xlim(dx)
...             axarr[j, i].set_ylim(dy)
...             axarr[j, i].set_xticks(dx)
...             axarr[j, i].set_yticks(dy)
...             plt.axis('equal')
>>> #             axarr[j, i].set_xlim([spot[i, j, 0].min(), spot[i, j, 0].max()])
... #             axarr[j, i].set_ylim([spot[i, j, 1].min(), spot[i, j, 1].max()])
...
...
... #             dx = min(spot[:, :, 0]) - max(spot[:, :, 0])
... #             dy = min(spot[:, :, 1]) - max(spot[:, :, 1])
...
... #             axarr[j, i].set_xlim([])
... #             axarr[j, i].set_ylim([])
...
... #             axarr[j, i].set_xlim([min(d2[-1,0,:]) - (max(d2[-1,0,:]) - min(d2[-1,0,:]))/2, max(d2[-1,0,:]) + (max(d2[-1,0,:]) - min(d2[-1,0,:]))/2])
... #             axarr[j, i].set_ylim([min(d2[-1,1,:]) - (max(d2[-1,1,:]) - min(d2[-1,1,:]))/2, max(d2[-1,1,:]) + (max(d2[-1,1,:]) - min(d2[-1,1,:]))/2])
...
...     fig.tight_layout()
...     plt.show()
...     return
...
...
>>> def plotRays(t, d, k):
...     # Adjust distance vector to include z-distance
...     distance = np.cumsum(t)
...     z_addition = np.vstack((np.zeros(3), np.outer(distance, k)))
...     z_addition = np.transpose(np.tile(z_addition, (len(d[0, 0, :]), 1 , 1)), axes = [1, 2, 0])
...     d2 = d + z_addition
...
...     # Plot all rays (not smart for 1000 rays!)
...     fig = plt.figure(figsize=(12, 8))
...     ax = fig.add_subplot(111, projection='3d')
...     for i in range(len(d[0, 0, :])):
...         ax.plot(d2[:, 1, i], d2[:, 2, i], d2[:, 0, i])
...     ax.set_xlabel('$y$')
...     ax.set_ylabel('$z$')
...     ax.set_zlabel('$x$')
>>> #     fig.tight_layout()
...     plt.show()
...     return
```

---
scrolled: true
...

```python
>>> # Index of refraction for the common N-BK7 (SCHOTT) glass at wavelengths:
... n = np.array([[1, 1.5224, 1, 1.5224, 1, 1.5224, 1], # 0.486 um
...               [1, 1.5187, 1, 1.5187, 1, 1.5187, 1], # 0.546 um
...               [1, 1.5143, 1, 1.5143, 1, 1.5143, 1]]) # 0.656 um
...
>>> Nsteps = len(n[0, :])
...
>>> bounds = [(1/1700, 1/300), (-1/300, -1/1700), (-1/300, -1/1700), (1/1700, 1/300), (1/1700, 1/300), (-1/300, -1/1700), (0, 0),
...           (0, 0), (0, 300), (0, 300), (0, 300), (0, 300), (0, 300), (1000, 1500)]
...
>>> result = differential_evolution(costFunction, bounds, maxiter = 1)
>>> print(result)
...
>>> fitnessFactor = costFunction(result.x, plotyn=True)
...
>>> print('\n fitnessFactor', fitnessFactor)
>>> print('\n R', 1/result.x[:Nsteps-1])
>>> print('\n t', result.x[Nsteps:])
     fun: 0.059585556463146103
 message: 'Maximum number of iterations has been exceeded.'
     jac: array([  1.21329955e+02,   1.15346319e+02,   6.22621611e+01,
         7.24377067e+01,   1.04332405e+02,   1.73248503e+02,
         3.38015033e-03,   9.29811783e-08,  -2.97255276e-05,
        -1.94136374e-05,  -4.14390744e-06,   1.27597932e-04,
         3.23761851e-05,   2.08381923e-05])
     nit: 1
       x: array([  2.06813341e-03,  -6.89156540e-04,  -2.90746915e-03,
         3.27338851e-03,   9.25958207e-04,  -2.97022778e-03,
         0.00000000e+00,   0.00000000e+00,   6.30480986e+01,
         6.87576176e+01,   1.45066585e+02,   7.72613330e+01,
         2.40637765e+02,   1.34002623e+03])
 success: False
    nfev: 780
D:\Program Files\Anaconda3\lib\site-packages\scipy\optimize\_differentialevolution.py:572: RuntimeWarning: invalid value encountered in true_divide
  return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5





 fitnessFactor 0.0595855564631

 R [  483.52780203 -1451.04913253  -343.94173999   305.49383224  1079.96234835
  -336.67451538]

 t [    0.            63.04809862    68.75761763   145.06658505    77.26133305
   240.63776463  1340.02623116]

<IPython.core.display.Javascript object>
<IPython.core.display.HTML object>
<IPython.core.display.Javascript object>
<IPython.core.display.HTML object>
```

```python

```

```python

```
