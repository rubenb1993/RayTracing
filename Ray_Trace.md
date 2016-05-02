# Ray Tracing and optimization of a Cooke-triplet

By Ruben Biesheuvel and Aloys Erkelens

This software looks to optimize the radii of curvature and distances between the lenses of a Cooke-Triplet in order to find the optimum configuration of these parameters to get the best focus for 3 different wavelengths.

The optimization is done by minimizing the spot size in the focal plane. As this is done for 3 wavelengths, the following expression is obtained for the spot size [1]:

$$ D = \left[ \frac{2}{3N(3N - 1)} \sum_{i<j} d_{i,j}^2  \right] ^{1/2}$$

A parallel beam of N rays is taken for each wavelength and each angle specified w.r.t. the optical axis. The spot size D is weighted per angle, counting the rays with no angle to the optical axis strongest.

Seeing that often CCDs are used in imaging, a straight image plane is taken for minimization.

# References:

[1] Van Leijenhorst, D. C., Lucasius, C. B., & Thijssen, J. M. (1996). Optical design with the aid of a genetic algorithm. $\textit{BioSystems}$, 37(3), 177-187.

The cost function is structured as follows:
    1. Initialze a beam of N parallel rays with angle theta
    2. Trace the rays through the whole optical system
    3. Determine the spot size acording to van Leijenhorst et al.
    4. Sum spot size over all wavelengths and all angles

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
...
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
...     u[0, :] = np.array([[angle], [0], [0]])
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
...
>>> def fitness(d):
...     '''Calculate the fitness of the spot in the image plane, according to the fitness factor introduced in
...     van Leijenhorst et al. (Biosystems, 1996)
...     input: d (either vector with images of all the planes or just the last plane).
...     Returns the scalar value for the fitness'''
...     N = len(d[0, 0, :]) # Number of rays
...     ind = np.triu_indices(N, k=1) #create array of upper triangular matrix indices
...
...     #calculate distance vector for i > j
...     dist = (d[-1, 1, ind[0]] - d[-1, 1, ind[1]])**2 + (d[-1, 0, ind[0]] - d[-1, 0, ind[1]])**2
...     sumDistance = np.sum(dist)
...     fitness = np.sqrt(2/(3*N*(3*N-1)) * sumDistance)
...
...     return fitness
...
...
>>> def costFunction(x,*args):
...     """The cost function that has to be minimized. Takes vector x composed of curvatures (c) and distances (t).
...     Additional arguments must be defined. These are: angles, weights, Rmax, Nr, Ntheta, n, plotyn.
...     See incomingBeam for the structure of Rmax, Nr and Ntheta.
...     angles is a vector with all the angles that have to be taken into account,
...     weights an equally long vector with their weights for the cost function.
...     n is the vector with all the refractive indices, and plotyn is a Boolean denoting whether or not to plot the results.
...     Returns a scalar fitnessFactor which is to be minimized."""
...
...     c = x[:Nsteps]
...     t = x[Nsteps:]
...
...     fitnessFactor = 0
...     spot = np.zeros((len(n), len(angles), 3, N_rays))
...
...     for i in range(len(n)):
...         for j in range(len(angles)):
...             d, u, k = incomingBeam(Rmax, Nr, Ntheta, angles[j])
...             d = traceRays(c, t, n[i], d, u, k, Nsteps)
...             spot[i, j] = d[-1]
...             fitnessFactor += weights[j] * fitness(d)
...
...     if plotyn == True:
...         plotSpot(spot, angles)
...         plotRays(t, d, k)
...
...     return fitnessFactor
...
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
...             plt.axis('square')
...
...     fig.tight_layout()
...
...     plt.show()
...     return
...
...
>>> def plotRays(t, d, k):
...     """A function to plot the results of all the beams at all the angles."""
...
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
...     plt.show()
...     return
```

---
scrolled: false
...

```python
>>> # Index of refraction for the common N-BK7 (SCHOTT) glass at wavelengths:
... n = np.array([[1, 1.5224, 1, 1.5224, 1, 1.5224, 1], # 0.486 um
...               [1, 1.5187, 1, 1.5187, 1, 1.5187, 1], # 0.546 um
...               [1, 1.5143, 1, 1.5143, 1, 1.5143, 1]]) # 0.656 um
...
>>> angles = np.array([0.00300, 0.00225, 0.00150, 0.00075, 0])
>>> weights = [0.125, 0.125, 0.125, 0.125, 0.5]
>>> Rmax = 0.1
>>> Nr = 6
>>> Ntheta = 25
>>> Nsteps = len(n[0, :])
>>> plotyn = False
...
>>> d, u, k = incomingBeam(Rmax, Nr, Ntheta, 0) # Make vector to predict amount of rays
>>> N_rays = len(d[0,0,:])
...
>>> bounds = [(1/1700, 1/300), (-1/300, -1/1700), (-1/300, -1/1700), (1/1700, 1/300), (1/1700, 1/300), (-1/300, -1/1700), (0, 0),
...           (0, 0), (0, 300), (0, 300), (0, 300), (0, 300), (0, 300), (2000, 2000)]
...
>>> result = differential_evolution(costFunction, bounds, args=(angles, weights, Rmax, Nr, Ntheta, n, plotyn), maxiter = 10)
>>> print(result)
...
>>> plotyn = True
>>> fitnessFactor = costFunction(result.x, angles, weights, Rmax, Nr, Ntheta, n, plotyn)
...
>>> print('\n fitnessFactor', fitnessFactor)
>>> print('\n R', 1/result.x[:Nsteps-1])
>>> print('\n t', result.x[Nsteps:])
>>> print('\n N_rays', len(d[0,0,:]))
```

Since the code does not converge to a global minimum and appears to be stuck in a local minimum, here would be some code to try to find the focus of the other 2 beams using the paraxialFocus function to estimate the focal point. Other optimizations as well as a different fitness function could be tried out, where a genetic algorithm is apparantly prone to finding local minima, and the fitness function might tend towards 1 spot in focus.
