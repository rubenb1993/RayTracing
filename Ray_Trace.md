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
...
>>> def paraxialFocus(c, n, t):
...     '''Determine the focus in the paraxial limit'''
...
...     f = sympy.Symbol('f')
...     t = sympy.Matrix(t)
...     t[-1] = f
...
...     product2 = np.identity(2)
...
...     for i in range(len(t)):
...         T = np.array([[1, t[i]], [0, 1]])
...
...         product1 = np.dot(T, product2)
...
...         if i == len(t)-1:
...             break
...
...         R = np.array([[1, 0], [c[i]*(n[i] - n[i+1])/n[i+1], n[i]/n[i+1]]])
...
...         product2 = np.dot(R, product1)
...
...     return sympy.solve(product1[0, 0], f)[0]
...
...
>>> def incomingBeam(Rmax = 2, Nr=4, Ntheta=25):
...     # Circular starting pattern (could also be set to random if needed)
...
...     r = np.array([np.linspace(0.5, Rmax, Nr) for i in range(Ntheta)])
...     theta = np.array([np.linspace(0, 2*np.pi, Ntheta) for i in range(Nr)]).T
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
...     u[0, :] = np.vstack(np.array([0, 0, 0]))
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
...
...     sumDistance = 0
...     N = len(d[0, 0, :]) # Number of rays
...
...     for i in range(N):
...         for j in range(i+1, N):
...             sumDistance += (d[-1, 1, i] - d[-1, 1, j])**2 + (d[-1, 2, i] - d[-1, 2, j])**2
...
...     fitness = np.sqrt(2/(3*N*(3*N-1)) * sumDistance)
...
...     return fitness
...
...
>>> def costFunction(x, plotyn=False):
...     middle = int(len(x)/2)
...     c = x[:Nsteps]
...     t = x[Nsteps:]
...
...     d, u, k = incomingBeam()
...     d = traceRays(c, t, n, d, u, k)
...     fitnessFactor = fitness(d)
...
...     if plotyn == True:
...         plotResults(t, d, k)
...
...     return fitnessFactor
...
...
>>> def plotResults(t, d, k):
...
...     # Adjust distance vector to include z-distance
...     distance = np.cumsum(t)
...     z_addition = np.vstack((np.zeros(3), np.outer(distance, k)))
...     z_addition = np.transpose(np.tile(z_addition, (len(d[0, 0, :]), 1 , 1)), axes = [1, 2, 0])
...     d2 = d + z_addition
...
...     # Plot all rays (not smart for 1000 rays!)
...     fig = plt.figure()
...     ax = fig.add_subplot(111, projection='3d')
...     for i in range(len(d[0, 0, :])):
...         ax.plot(d2[:, 1, i], d2[:, 2, i], d2[:, 0, i])
...     ax.set_xlabel('$y$')
...     ax.set_ylabel('$z$')
...     ax.set_zlabel('$x$')
...     plt.show()
...
...     return
```

---
scrolled: true
...

```python
>>> # c = np.array([1/1204.9085, 1/-487.3933, 1/-496.7340, 1/-2435.0087, 0])
... # t = np.array([0.0965, 87.2903, 87.4564, 157.1440, 0])
... # t[-1] = paraxialFocus(c, n, t)
...
... n = np.array([1,  1.4866, 1, 1.5584, 1])
>>> Nsteps = len(n)
...
>>> bounds = [(1/300, 1/1700), (-1/300, -1/1700), (-1/1700, -1/300), (-1/1700, -1/300), (0, 0),
...           (0, 0), (0, 100), (0, 100), (0, 100), (1200, 1200)]
...
>>> result = differential_evolution(costFunction, bounds, maxiter=10)
>>> print(result)
...
>>> fitnessFactor = costFunction(result.x, plotyn=True)
...
>>> print('\n fitnessFactor', fitnessFactor)
>>> print('\n R', 1/result.x[:len(n)])
>>> print('\n t', result.x[len(n):])
     nit: 10
     fun: array(0.0)
 message: 'Maximum number of iterations has been exceeded.'
     jac: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
 success: False
       x: array([  1.41879633e-03,  -1.15545885e-03,  -2.11170926e-03,
        -1.16644022e-03,   0.00000000e+00,   0.00000000e+00,
         1.60520384e+01,   5.98715164e+01,   4.10803636e+01,
         1.20000000e+03])
    nfev: 1650
D:\Program Files\Anaconda3\lib\site-packages\scipy\optimize\_differentialevolution.py:572: RuntimeWarning: invalid value encountered in true_divide
  return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5



 fitnessFactor 2.04583005114e-05

R [ 704.82279934 -865.45704628 -473.55003696 -857.30926118           inf]

t [    0.            16.05203835    59.87151638    41.08036357  1200.        ]
D:\Program Files\Anaconda3\lib\site-packages\ipykernel\__main__.py:17: RuntimeWarning: divide by zero encountered in true_divide

<IPython.core.display.Javascript object>
<IPython.core.display.HTML object>
```

```python

```

```python

```

```python
>>> n = np.array([1,  1.4866, 1, 1.5584, 1])
>>> c = np.array([1/1204.9085, 1/-487.3933, 1/-496.7340, 1/-2435.0087, 0])
>>> t = np.array([0.0965, 87.2903, 87.4564, 157.1440, 0])
>>> t[-1] = paraxialFocus(c, n, t)
...
>>> Nsteps = len(n)
...
>>> d, u, k = incomingBeam()
>>> d = traceRays(c, t, n, d, u, k)
>>> fitnessFactor = fitness(d)
...
>>> # Adjust distance vector to include z-distance
... distance = np.cumsum(t)
>>> z_addition = np.vstack((np.zeros(3), np.outer(distance, k)))
>>> z_addition = np.transpose(np.tile(z_addition, (len(d[0, 0, :]), 1 , 1)), axes = [1, 2, 0])
>>> d2 = d + z_addition
...
>>> # Plot all rays (not smart for 1000 rays!)
... fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> for i in range(len(d[0, 0, :])):
...     ax.plot(d2[:, 1, i], d2[:, 2, i], d2[:, 0, i])
>>> ax.set_xlabel('$y$')
>>> ax.set_ylabel('$z$')
>>> ax.set_zlabel('$x$')
>>> plt.show()
<IPython.core.display.Javascript object>
<IPython.core.display.HTML object>
```

```python
>>> %matplotlib inline
>>> #plot a scatter plot through the focus
... plt.scatter(d[-3, 0, :], d[-3, 1, :], color = 'r')
>>> plt.scatter(d[-2, 0, :], d[-2, 1, :], color = 'c')
>>> plt.scatter(d[-1, 0, :], d[-1, 1, :])
>>> plt.show()
```

```python

```
