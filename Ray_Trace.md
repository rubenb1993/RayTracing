```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import Axes3D
>>> from math import pi
...
>>> %matplotlib inline

```

```python
>>> def pol2cart(rho, phi):
...     "converts polar to cartesian coordinates"
...     x = rho * np.cos(phi)
...     y = rho * np.sin(phi)
...     return[x, y]
```

```python
>>> #Circular starting pattern (could also be set to random if needed)
... r = np.array([[0.5, 1, 1.5, 2] for i in range(25)])
>>> theta = np.array([np.linspace(0,2*pi,25) for i in range(4)]).T
>>> [X, Y] = pol2cart(r,theta)
>>> # plt.scatter(X,Y)
... # plt.figaspect(1)
...
... X = X.reshape(-1)
>>> Y = Y.reshape(-1)
...
>>> # X= [1, 0, -1]
... # Y = [1, 0, -1]
...
... # #Random starting pattern (could also be set to circular if needed)
... # X = 3*(np.random.random(size = (100)) - 0.5)
... # Y = 3*(np.random.random(size = (100)) - 0.5)
```

```python
>>> #c = np.array([0.1, -0.2, 0, -0.5, 0.8, 0, 0.001, -0.002])
... r = np.array([1204.9085, -487.3933, -496.7340, -2435.0087])
>>> c = 1/r
>>> c= np.append(c,[0])
>>> #c = np.array([0.01, -0.02, -0.05, 0.08, 0.08, -0.08, 0])
... n = np.array([1,  1.4866, 1, 1.5584, 1])
>>> t = np.array([0.0965, 87.2903, 87.4564, 157.1440, 0])
>>> t[-1] = 2000 - np.cumsum(t)[-1]
...
>>> #define unit vector in z-dimonsion
... k = np.array([0, 0, 1])
>>> k_trans = np.array([0, 0, 1]) #Copy k manually due to k also being transposed if k_trans = k
>>> k_trans.shape = (3,1)
...
>>> #position and angle vector (place,coordinate,beam#)
... d = np.zeros(shape = (len(t)+1, 3, len(X)))
>>> d[0,0] = X
>>> d[0,1] = Y
>>> u = np.zeros(shape = (len(t)+1, 3, len(X)))
>>> u[0, :] = np.vstack(np.array([0, 0, 0]))
>>> u[0,2] = np.sqrt(1 - u[0,0]**2 - u[0,1]**2) #calculate angle with z-axis based on x and y axes
```

```python
>>> c = np.array([0.05, -0.05, 0, 0, 0, 0, 0, 0])
>>> n = np.array([1, 1.5, 1, 1, 1, 1, 1, 1])
>>> t = np.array([10, 1, 19.8])
...
>>> #position and angle vector (place,coordinate,beam#)
... d = np.zeros(shape = (len(t)+1, 3, len(X)))
>>> d[0,0] = X
>>> d[0,1] = Y
>>> u = np.zeros(shape = (len(t)+1, 3, len(X)))
>>> u[0, :] = np.vstack(np.array([0, 0, 0]))
>>> u[0,2] = np.sqrt(1 - u[0,0]**2 - u[0,1]**2) #calculate angle with z-axis based on x and y axes
```

```python
>>> #Compute the ray paths
... for i in range(len(t)):
...     #Algorithm to obtain new positions
...     e = t[i]*np.dot(k, u[i]) - np.sum(d[0]*u[0], axis = 0) #a scalar for obtaining the total travel distance T
...     #Mz = np.dot(k, d[i]) + e*np.dot(k, u[i]) - t[i] #z component of M vector
...     M = d[i] + e*u[i] - t[i]*k_trans
...     Mz = M[2]
...     M2 = np.sum(M*M, axis = 0)
...     cos_inc_angle = np.sqrt(u[i,2]**2 - c[i]*(c[i]*M2 - 2*Mz))
...     T = e + (c[i]*M2 - 2*Mz)/(u[i,2] + cos_inc_angle) #total path length to interface
...     #Translate d over distance T. Remove z-component to be in the plane perpendicular to z at point z=ti
...     d[i+1] = d[i] + T*u[i] - t[i]*k_trans
...
...     #At the last plane, don't calculate angles
...     if i == len(t)-1:
...         break
...
...     #Algorithm to calculate new angle due to refraction
...     mu = n[i]/n[i+1]
...     cos_out_angle = np.sqrt(1 - mu**2 * (1 - cos_inc_angle**2))
...     g = cos_out_angle - mu*cos_inc_angle
...     u[i+1] = mu*u[i] - c[i]*g*d[i+1] + g*k_trans
...
>>> #Adjust distance vector to include z-distance
... distance = np.cumsum(t)
>>> z_addition = np.vstack((np.zeros(3),np.outer(distance,k)))
>>> z_addition = np.transpose(np.tile(z_addition,(len(d[0,0,:]),1,1)),axes = [1,2,0])
>>> d = d + z_addition
```

---
scrolled: false
...

```python
>>> #plot all rays (not smart for 1000 rays!)
... fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> for i in range(len(X)):
...     ax.plot(d[:,1,i],d[:,2,i],d[:,0,i])
...
>>> ax.set_xlabel('X label')
>>> ax.set_ylabel('Y Label')
>>> ax.set_zlabel('Z Label')
```

```python
>>> #plot a scatter plot through the focus
... plt.scatter(d[-3, 0, :], d[-3, 1, :], color = 'r')
>>> plt.scatter(d[-2, 0, :], d[-2, 1, :], color = 'c')
>>> plt.scatter(d[-1, 0, :], d[-1, 1, :])
```
