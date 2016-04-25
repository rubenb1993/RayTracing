```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import Axes3D
...
>>> %matplotlib notebook
```

```python
>>> c = np.array([0.05, -0.1,0])
>>> n = np.array([1, 1.5, 1])
>>> t = np.array([10, 5, 12])
...
>>> i = np.array([1, 0, 0])
>>> j = np.array([0, 1, 0])
>>> k = np.array([0, 0, 1])
...
>>> d = np.zeros(shape = (len(t)+1, 3))
>>> d[0] = np.array([2, 1, 2])
>>> u = np.zeros(shape = (len(t)+1, 3))
>>> u[0] = np.array([0.2, 0.2, 0])
>>> u[0,2] = np.sqrt(1 - u[0,0]**2 - u[0,1]**2)
...
>>> for i in range(len(t)):
...     #Algorithm to obtain new positions
...     e = t[i]*np.dot(k, u[i]) - np.dot(d[i], u[i]) #a scalar for obtaining the total travel distance T
...     Mz = np.dot(d[i], k) + e*np.dot(u[i], k) - t[i] #z component of M vector
...     M = d[i] + e*u[i] - t[i]*k
...     M2 = np.dot(M, M)
...     cos_inc_angle = np.sqrt(u[i,2]**2 - c[i]*(c[i]*M2 - 2*Mz))
...     T = e + (c[i]*M2 - 2*Mz)/(u[i,2] + cos_inc_angle) #total path length to interface
...     d[i+1] = d[i] + T*u[i] - t[i]*k
...
...     if i == len(t)-1:
...         break
...
...     mu = n[i]/n[i+1]
...     cos_out_angle = np.sqrt(1 - mu**2 * (1 - cos_inc_angle**2))
...     g = cos_out_angle - mu*cos_inc_angle
...     u[i+1] = mu*u[i] - c[i]*g*d[i+1] + g*k
...
>>> distance = np.cumsum(t)
>>> z_addition = np.vstack((np.zeros(3),np.outer(distance,k)))
>>> d = d + z_addition
>>> print(d)
```

---
scrolled: true
...

```python
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> ax.plot(d[:,1],d[:,2],d[:,0])
>>> ax.set_xlabel('X label')
>>> ax.set_ylabel('Y Label')
>>> ax.set_zlabel('Z Label')
```

```python

```
