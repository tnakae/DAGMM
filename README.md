# DAGMM Tensorflow implementation
Deep Autoencoding Gaussian Mixture Model.

This implementation is based on the paper
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]](https://openreview.net/pdf?id=BJJLHbb0-)

this is UNOFFICIAL implementation.

# Requirements
- python (3.5-3.6)
- Tensorflow <= 1.15
- Numpy
- sklearn

# Usage instructions
To use DAGMM model, you need to create "DAGMM" object.
At initialize, you have to specify next 4 variables at least.

- ``comp_hiddens`` : list of int
  - sizes of hidden layers of compression network
  - For example, if the sizes are ``[n1, n2]``,
  structure of compression network is:
  ``input_size -> n1 -> n2 -> n1 -> input_sizes``
- ``comp_activation`` : function
  - activation function of compression network
- ``est_hiddens`` : list of int
  - sizes of hidden layers of estimation network.
  - The last element of this list is assigned as n_comp.
  - For example, if the sizes are ``[n1, n2]``,
    structure of estimation network is:
    ``input_size -> n1 -> n2 (= n_comp)``
- ``est_activation`` : function
  - activation function of estimation network

Then you fit the training data, and predict to get energies
(anomaly score). It looks like the model interface of scikit-learn.

For more details, please check out [dagmm/dagmm.py](dagmm/dagmm.py) docstrings.

# Example
## Small Example
``` python
import tensorflow as tf
from dagmm import DAGMM

# Initialize
model = DAGMM(
  comp_hiddens=[32,16,2], comp_activation=tf.nn.tanh,
  est_hiddens=[16.8], est_activation=tf.nn.tanh,
  est_dropout_ratio=0.25
)
# Fit the training data to model
model.fit(x_train)

# Evaluate energies
# (the more the energy is, the more it is anomary)
energy = model.predict(x_test)

# Save fitted model to the directory
model.save("./fitted_model")

# Restore saved model from dicrectory
model.restore("./fitted_model")
```

## Jupyter Notebook Example
You can use next jupyter notebook examples using DAGMM model.
- [Simple DAGMM Example notebook](Example_DAGMM.ipynb) :
This example uses random samples of mixture of gaussian.
If you want to know simple usage, this notebook is recommended.
- [KDDCup99 10% Data Evaluation](KDDCup99.ipynb) :
Performance evaluation of anomaly detection for KDDCup99 10% Data
with the same condition of original paper (need pandas)

# Notes
## GMM Implementation
The equation to calculate "energy" for each sample in the original paper
uses direct expression of multivariate gaussian distribution which
has covariance matrix inversion, but it is impossible sometimes
because of singularity.

Instead, this implementation uses cholesky decomposition of covariance matrix.
(this is based on [GMM code in Tensorflow code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/factorization/python/ops/gmm_ops.py))

In ``DAGMM.fit()``, it generates and stores triangular matrix of cholesky decomposition
of covariance matrix, and it is used in ``DAGMM.predict()``,

In addition to it, small perturbation (1e-6) is added to diagonal
elements of covariance matrix for more numerical stability
(it is same as Tensorflow GMM implementation,
and [another author of DAGMM](https://github.com/danieltan07/dagmm) also points it out)

## Parameter of GMM Covariance (lambda_2)
Default value of lambda_2 is set to 0.0001 (0.005 in original paper).
When lambda_2 is 0.005, covariances of GMM becomes too large to detect
anomaly points. But perhaps it depends on distribution of data and method of preprocessing
(for example a method of normalization). Recommend to control lambda_2
when performance metrics is not good.
