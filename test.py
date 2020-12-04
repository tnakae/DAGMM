import tensorflow as tf
import numpy as np
from sklearn.datasets import make_blobs

from dagmm import DAGMM

def main():
    data, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=123)

    data[300] = [-1, -1, -1, -1, -1]
    data[500] = [ 1,  0,  1,  1,  1]
    ano_index = [300, 500]

    tf.reset_default_graph()
    model_dagmm = DAGMM(
        comp_hiddens=[16,8,1], comp_activation=tf.nn.tanh,
        est_hiddens=[8,4], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
        epoch_size=1000, minibatch_size=128
    )

    model_dagmm.fit(data)
    energy = model_dagmm.predict(data)
    print("fitted_energy = {:.3f}".format(energy.mean()))

if __name__ == "__main__":
    main()

