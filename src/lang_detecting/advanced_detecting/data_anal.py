from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import operator as op

# (len, n_?) => recognizability
predictabilities = {
    (3, 1): 0,
    (3, 2): 0,
    (4, 1): .5,
    (4, 2): 0,
    (5, 1): .5,
    (5, 2): .5,

    (6, 2): .7,
    (6, 3): 0,
    (7, 2): 1,
    (7, 3): 0,
    (8, 2): 1,
    (8, 3): .7,
    (9, 2): 1,
    (9, 3): 1,
}


x_vals = [k[0] for k in predictabilities.keys()]
y_vals = [k[1] for k in predictabilities.keys()]
z_vals = list(predictabilities.values())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Representation of Predictability')
ax.set_xlabel('Length'); ax.set_ylabel('Uniq'); ax.set_zlabel('Predictability')
ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis', s=100)

import numpy as np

# Make a grid for surface
x_unique, y_unique = sorted(set(x_vals)), sorted(set(y_vals))
X, Y = np.meshgrid(x_unique, y_unique)
Z = np.zeros_like(X, dtype=float)
for (i, x), (j, y) in product(enumerate(x_unique), enumerate(y_unique)):
    Z[j, i] = predictabilities.get((x, y), np.nan)

ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='k')
#plt.show()

def visualize_relation(func: Callable, title: str = 'Pred', xlabel: str = 'x', *, data: dict[tuple[int, int], float] = None):
    data = data or predictabilities
    func_repr = [(func(l, u), u, p) for ((l, u), p) in data.items()]
    l_sorted, u_sorted, p_sorted = zip(*sorted(func_repr))
    plt.figure(figsize=(8, 5))
    plt.plot(l_sorted, p_sorted, marker='o', linestyle='-', color='blue')
    for l, u_val, u_label in zip(l_sorted, p_sorted, u_sorted):
        plt.text(l, u_val + 0.03, str(u_label), ha='center', va='bottom', fontsize=9)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel('Predictability')
    plt.grid(True)

visualize_relation(op.sub, 'Predictability vs L-U', 'Length - Unique')
visualize_relation(op.truediv, 'Predictability vs L/U', 'L/U')
visualize_relation(lambda l, u: (l-3)/u, 'Predictability vs L-3/U', 'L-3/U')
plt.show()
