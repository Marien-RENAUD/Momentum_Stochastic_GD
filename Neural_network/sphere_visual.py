import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.patches as mpatches
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Sphere data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
color = np.zeros((100,100),dtype=int)

# Plot the surface

color = np.where(z > 0, 300, 0)  # Assign 1 to positive z, 0 to negative z
color_map = cm.coolwarm(color)  # Use a colormap based on this data
ax.plot_surface(x, y, z, facecolors=color_map, rstride=1, cstride=1,alpha = 0.6)
# Set an equal aspect ratio
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_zticklabels([])
ax.set_zticks([])
red_patch = mpatches.Patch(color='red', label='label 1')
blue_patch = mpatches.Patch(color='blue', label='label 0')
plt.legend(handles=[blue_patch,red_patch])
# Remove gridlines
ax.grid(False)
plt.show()