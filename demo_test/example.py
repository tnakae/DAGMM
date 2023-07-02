import tensorflow as tf
import numpy as np
from sklearn.datasets import make_blobs

from dagmm import DAGMM

import matplotlib.pyplot as plt

data, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=123)

data[300] = [-1, -1, -1, -1, -1]
data[500] = [1, 0, 1, 1, 1]
ano_index = [300, 500]

plt.figure(figsize=[8, 8])
plt.plot(data[:, 0], data[:, 1], ".")
plt.plot(data[ano_index, 0], data[ano_index, 1], "o", c="r", markersize=10)
plt.show()

tf.reset_default_graph()
model_dagmm = DAGMM(
    comp_hiddens=[16, 8, 1], comp_activation=tf.nn.tanh,
    est_hiddens=[8, 4], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
    epoch_size=1000, minibatch_size=128
)
model_dagmm.fit(data)
energy = model_dagmm.predict(data)

plt.figure(figsize=[8, 3])
histinfo = plt.hist(energy, bins=50)
plt.xlabel("DAGMM Energy")
plt.ylabel("Number of Sample(s)")
plt.show()

plt.figure(figsize=[8, 3])
plt.plot(energy, "o-")
plt.xlabel("Index (row) of Sample")
plt.ylabel("Energy")
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=[12, 12], sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for row in range(5):
    for col in range(5):
        ax = axes[row, col]
        if row != col:
            ax.plot(data[:, col], data[:, row], ".")
            ano_index = np.arange(len(energy))[energy > np.percentile(energy, 99)]
            ax.plot(data[ano_index, col], data[ano_index, row], "x", c="r", markersize=8)
plt.show()
