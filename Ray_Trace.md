```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import Axes3D
>>> from scipy.optimize import differential_evolution
>>> import sympy
>>> import timeit
...
>>> %matplotlib inline
```

```python
>>> def pol2cart(rho, phi):
...     '''converts polar to cartesian coordinates
...     rho and phi are both scalars, or both vectors with same size.
...     Returns 2 arrays x and y with the same size as rho and phi'''
...     x = rho * np.cos(phi)
...     y = rho * np.sin(phi)
...     return x, y
...
...
>>> def paraxialFocus(c, n, t):
...     '''Determine the focus in the paraxial limit by working with the matrix convention.
...     Returns the scalar distance f from the last lens in order to be in focus'''
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
...     """Initiliazes the incoming beam.
...     Creates 5*(1+row#) points equally spaced points per row, with Nr rows and radius Rmax/Nr * (i+1).
...     angle is used to set the initial angle with the x-axis, in terms of cos(angle).
...     Returns d \in (len(t), 3, N) (the position matrix at each plane),
...     Returns u \in (len(t), 3, N) (the angle matrix at each plane),
...     Returns k \in (3,) (the normalized vector in the direction of the optical axis)"""
...     # Circular starting pattern (could also be set to random if needed)
...
...     r = np.array([])
...     theta = np.array([])
...     for i in range(Nr):
...         R = (i+1)*Rmax/Nr
...         j = (i+1)*5 #increase of 5 points per row
...         a = np.array([R]*j)
...         r = np.append(r,a)
...         b = np.linspace(0, 2*np.pi, j, endpoint = False) #to not include 2pi in the end point.
...         theta = np.append(theta, b)
...     X, Y = pol2cart(r, theta)
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
...     u[0, :] = np.vstack(np.array([angle, 0, 0]))
...     u[0, 2] = np.sqrt(1 - u[0, 0]**2 - u[0, 1]**2) # Calculate angle with z-axis based on x and y axes
...
...     return d, u, k
...
...
>>> def traceRays(c, t, n, d, u, k, Nsteps):
...     '''Compute the ray paths through each optical element.
...     Algorithm worked out by W. Bruce Zimmerman, Ph.D, Indiana University South Bend (http://www.learnoptics.com/).
...     Inputs: c (vector of reciprocal radii of optical elements), t (vector of distances between optical elements),
...     n (vector of indices of refraction of each optical element), d (initialized position vector (see incomingBeam)),
...     u (initialzed angle vector (see incomingBeam)), k (normalized vector along the optical axis),
...     Nsteps (amount of optical planes)
...     Returns d: total position vector for each ray with respect to each plane (Nsteps, 3, N_rays).'''
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
>>> def fitness(d):
...     '''Calculate the fitness of the spot in the image plane, according to the fitness factor introduced in
...     van Leijenhorst et al. (Biosystems, 1996)
...     input: d (either vector with images of all the planes or just the last plane).
...     Returns the scalar value for the fitness'''
...     N = len(d[0, 0, :]) # Number of rays
...     ind = np.triu_indices(N, k=1) #create array of upper triangular matrix indices
...     #calculate distance vector for i > j
...     dist = (d[-1, 1, ind[0]] - d[-1, 1, ind[1]])**2 + (d[-1, 0, ind[0]] - d[-1, 0, ind[1]])**2
...     sumDistance = np.sum(dist)
...     fitness = np.sqrt(2/(3*N*(3*N-1)) * sumDistance)
...     return fitness
...
>>> def costFunction(x, *args):
...     """The cost function that has to be minimized. Takes vector x composed of curvatures (c) and distances (t).
...     Additional arguments should be defined. These are: angles, weights, Rmax, Nr, Ntheta, n, plotyn.
...     See incomingBeam for the structure of Rmax, Nr and Ntheta.
...     angles is a vector with all the angles that have to be taken into account,
...     weights an equally long vector with their weights for the cost function.
...     n is the vector with all the refractive indices, and plotyn is a Boolean denoting whether or not to plot the results.
...     Returns a scalar fitnessFactor which is to be minimized."""
...     Nsteps = len(n)
...     c = x[:Nsteps]
...     t = x[Nsteps:]
...     fitnessFactor = 0
...     for j in range(len(angles)):
...         d, u, k = incomingBeam(Rmax, Nr, Ntheta, angles[j])
...         d = traceRays(c, t, n, d, u, k, Nsteps)
...         fitnessFactor += weights[j] * fitness(d)
...         if plotyn == True:
...             plotResults(t, d, k)
...
...     return fitnessFactor
...
...
>>> def plotResults(t, d, k):
...     """A function to plot the results of all the beams at all the angles.
...     Aloys will make changes so I don't touch this."""
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
...     #fig2 = plt.figure()
...     ax = plt.subplot(111)
...     ax.scatter(d2[-1,0,:], d2[-1,1,:])
...     #axes = plt.gca()
...     ax.set_xlim([min(d2[-1,0,:]) - (max(d2[-1,0,:]) - min(d2[-1,0,:]))/2, \
...                  max(d2[-1,0,:]) + (max(d2[-1,0,:]) - min(d2[-1,0,:]))/2])
...     ax.set_ylim([min(d2[-1,1,:]) - (max(d2[-1,1,:]) - min(d2[-1,1,:]))/2, \
...                  max(d2[-1,1,:]) + (max(d2[-1,1,:]) - min(d2[-1,1,:]))/2])
...     plt.show()
...     return
```

---
scrolled: true
...

```python
>>> n = np.array([1,  1.4866, 1, 1.5584, 1])
>>> angles= np.sin([0, 0.0075, 0.0150, 0.0225, 0.0300])/10
>>> weights = [0.5, 0.125, 0.125, 0.125, 0.125]
>>> Rmax = 5
>>> Nr = 4
>>> Ntheta = 25
>>> plotyn = False
...
>>> bounds = [(1/300, 1/1700), (-1/300, -1/1700), (-1/1700, -1/300), (-1/1700, -1/300), (0, 0),
...           (0, 0), (0, 300), (0, 300), (0, 300), (300, 1500)]
...
>>> result = differential_evolution(costFunction, bounds, args=(angles, weights, Rmax, Nr, Ntheta, n, plotyn), maxiter = 10)
>>> print(result)
...
>>> plotyn = True
>>> fitnessFactor = costFunction(result.x, angles, weights, Rmax, Nr, Ntheta, n, plotyn)
...
>>> print('\n fitnessFactor', fitnessFactor)
>>> print('\n R', 1/result.x[:len(n)-1])
>>> print('\n t', result.x[len(n):])
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
```

```python
>>> %matplotlib inline
>>> #plot a scatter plot through the focus
... #plt.scatter(d[-3, 0, :], d[-3, 1, :], color = 'r')
... #plt.scatter(d[-2, 0, :], d[-2, 1, :], color = 'c')
... plt.scatter(d[-1, 0, :], d[-1, 1, :])
>>> plt.show()
```

```python
>>> ind = np.triu_indices(5, k=1)
>>> print(ind)
```

```python
>>> timeit.timeit(fitness(d))
```

```python
>>> %%timeit
... fitness(d)
```

```python
>>> %%timeit
... fitness_vector(d)
```

```python
>>> fig2 = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.scatter(d2[-1,0,:], d2[-1,1,:])
>>> axes = plt.gca()
>>> print(d2.shape)
>>> #axes.set_xlim([min(d2[-1,0,:])*1.5, max(d2[-1,0,:])*1.5])
... #axes.set_ylim([min(d2[-1,1,:])*1.5, max(d2[-1,1,:])*1.5])
```

```python
>>> plt.scatter(d2[0,0,:], d2[0,1,:])
```

```python
>>> print(min(d2[-1,0,:])*1.5)
>>> print(max(d2[-1,0,:])*1.5)
>>> print(min(d2[-1,1,:])*1.5)
>>> print(max(d2[-1,1,:])*1.5)
```

```python
>>> np.sin([0, 0.0075, 0.0150, 0.0225, 0.0300])
```

```python
>>> np.triu_indices(3, k=1)
```

```python
>>> d[-1, 2, ind[0]]
```

```python

```

```python

```
