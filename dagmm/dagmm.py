import tensorflow as tf

from dagmm.compression_net import CompressionNet
from dagmm.estimation_net import EstimationNet
from dagmm.gmm import GMM

class DAGMM:
    """ Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """
    def __init__(self, comp_hiddens, comp_activation,
            est_hiddens, est_activation, est_dropout_ratio=0.5,
            minibatch_size=1024, epoch_size=100,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.005):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        """
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)
        self.est_net = EstimationNet(est_hiddens, est_activation)
        self.est_dropout_ratio = est_dropout_ratio

        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Create tensorflow session
        self.sess = tf.InteractiveSession()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def fit(self, x):
        """ Fit the DAGMM model according to the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape

        # Create Placeholder
        self.input = input = tf.placeholder(
            dtype=tf.float32, shape=[None, n_features])
        self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[])

        # Build graph
        z, x_dash  = self.comp_net.inference(input)
        gamma = self.est_net.inference(z, drop)
        self.gmm.fit(z, gamma)
        energy = self.gmm.energy(z)

        self.x_dash = x_dash

        # Loss function
        loss = (self.comp_net.reconstruction_error(input, x_dash) +
            self.lambda1 * tf.reduce_mean(energy) +
            self.lambda2 * self.gmm.cov_diag_loss())

        # Minimizer
        minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # Number of batch
        n_batch = (n_samples - 1) // self.minibatch_size + 1

        # Create tensorflow session and initilize
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training
        for epoch in range(self.epoch_size):
            for batch in range(n_batch):
                i_start = batch * self.minibatch_size
                i_end = (batch + 1) * self.minibatch_size
                x_batch = x[i_start:i_end]

                self.sess.run(minimizer, feed_dict={
                    input:x_batch, drop:self.est_dropout_ratio})

            if (epoch + 1) % 100 == 0:
                loss_val = self.sess.run(loss, feed_dict={input:x, drop:0})
                print(f" epoch {epoch+1}/{self.epoch_size} : loss = {loss_val:.3f}")

        # Fix GMM parameter
        fix = self.gmm.fix_op()
        self.sess.run(fix, feed_dict={input:x, drop:0})
        self.energy = self.gmm.energy(z)

    def predict(self, x):
        """ Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        energies = self.sess.run(self.energy, feed_dict={self.input:x})
        return energies
