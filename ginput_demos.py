import time

import numpy as np
import matplotlib.pyplot as plt


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

plt.clf()
plt.axis([-1., 1., -1., 1.])
plt.setp(plt.gca(), autoscale_on=False)

tellme('You will define a triangle, click to begin')

plt.waitforbuttonpress()


while True:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')
        pts = np.asarray(plt.ginput(3, timeout=-1))
        if len(pts) < 3:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break

    # Get rid of fill
    for p in ph:
        p.remove()


# Define a nice function of distance from individual pts
def f(x, y, pts):
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1/(np.sqrt((x - p[0])**2 + (y - p[1])**2))
    return 1/z


X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
Z = f(X, Y, pts)

CS = plt.contour(X, Y, Z, 20)

tellme('Use mouse to select contour label locations, middle button to finish')
CL = plt.clabel(CS, manual=True)


tellme('Now do a nested zoom, click to begin')
plt.waitforbuttonpress()

while True:
    tellme('Select two corners of zoom, middle mouse button to finish')
    pts = np.asarray(plt.ginput(2, timeout=-1))

    if len(pts) < 2:
        break

    pts = np.sort(pts, axis=0)
    plt.axis(pts.T.ravel())

tellme('All Done!')
plt.show()

# library and dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Create data
df = pd.DataFrame({'x': range(1, 101), 'y': np.random.randn(100) * 15 + range(1, 101),
                   'z': (np.random.randn(100) * 15 + range(1, 101)) * 2})

# plot with matplotlib
plt.plot('x', 'y', data=df, marker='o', color='mediumvioletred')
plt.show()

# Just load seaborn and the chart looks better:
import seaborn as sns
sns.plot('x', 'y', data=df, marker='o', color='mediumvioletred')
sns.show()
