"""
Multivariate Maximum Entropy Neural Network Probability Density Function model

This file provides the necessary elements to replicate the multivariate tests of the paper.
"""

import numpy as np
import scipy.stats as st
import tensorflow as tf


def sobolInt(powDegree, lowerCorner, higherCorner):
    """
    Provides points and associated weights uniformly sampled by a Sobol' Design of Experiment
    
    Parameters
    ----------
    powDegree : int
        Power of 2 to use for the number of samples
    lowerCorner, higherCorner : array_like
        Lower and upper corners (len(lowerCorner),) of the hyperpercube sampled
    
    Returns
    -------
    array_like of float, array_like of float
        The array of the integration points (len(lowerCorner), 2**powDegree), and the array of the associated weights (2**powDegree,)
    """
    dim = len(lowerCorner)
    
    sobolSamp = st.qmc.Sobol(dim, scramble=False).random_base2(powDegree)
    weights =  (higherCorner - lowerCorner).prod() * (1. / 2 ** powDegree) * np.ones((sobolSamp.shape[0], 1))
    sobolSamp = (higherCorner - lowerCorner) * sobolSamp + lowerCorner

    return sobolSamp.astype(np.float32), weights.astype(np.float32)


def gaussianKdeInt(n, inputSample, bandwidth):
    """
    Provides points and associated weights sampled by a Gaussian KDE

    With a KDE lazily parametrized, the integration points density follows the sample density and avoids areas far from any points
    
    Parameters
    ----------
    n : int
        Number of samples desired
    inputSample : array_like
        Training sample (# of dims, # of data) used to generate the Gaussian KDE
    bandwidth : float
        Bandwidth of the Gaussian KDE
    
    Returns
    -------
    array_like of float, array_like of float
        The array of the integration points (# of dims, n), and the array of the associated weights (n,)
    """

    kde = st.gaussian_kde(inputSample, bandwidth)
    sampleKde = kde.resample(n)
    weightKde = 1. / (kde.pdf(sampleKde) * n)

    return sampleKde, weightKde


class SimpleMultiMaxEntrLayer(tf.keras.layers.Layer):
    """
    Layer enforcing the maximum entropy distribution formalism
    """

    def __init__(self, intPts, intWeights):
        """
        Parameters
        ----------
        intPts : array_like
            Points (# of data, # of dims) used to evaluate the integral
        intWeights : array_like, default: None
            Associated weights (1, # of data) of the points, if none are provided unifom weights are used
        """
        
        super(SimpleMultiMaxEntrLayer, self).__init__()
        self.mu = self.add_weight(shape=(), trainable=True, name="mu")
        self.lambda_ = self.add_weight(shape=(), trainable=False, name="lambda")

        if intWeights is None:
            nbPts = tf.shape(intPts)[0]
            intWeights = np.full((1, nbPts), 1./nbPts)

        self.logWeights = tf.constant(
            tf.math.log(intWeights),
            dtype="float32",      
            name='log weights'
            )
        self.intPts = tf.constant(
            intPts,
            dtype="float32",           
            name="integration points"
            )


    def call(self, inputs):
        """
        Defines the outputs when the layer is called
        
        Parameters
        ----------
        inputs: array_like
            Scalar outputs (# of data,) of a batch of evaluations of the neural network constraint estimation
        
        Returns
        -------
        tf.Tensor
            The estimated PDF values (# of data) for the given constraint evaluations
        """

        exponents = tf.math.add(
            tf.math.scalar_mul(self.mu, inputs),
            self.lambda_
        )
        return tf.math.exp(-exponents)[:, 0]
    
  
    def lambda_update(self, intConsOut):
        """
        Update the integration constant of the estimated PDF
        
        Parameters
        ----------
        intConsOut: array_like
            Evaluation of the constraint (# of data,) for the integration set

        Returns
        -------
        tf.Tensor
            Updated value of lambda (1,)
        """

        lambda_ = tf.math.reduce_logsumexp(
            tf.math.subtract(
                    self.logWeights,
                    tf.math.scalar_mul(self.mu, intConsOut)
                )
            )
        self.lambda_.assign(lambda_)
        return lambda_


class SimpleMultiMaxEntrModel(tf.keras.Model):
    """
    Model enforcing the maximum entropy distribution formalism
    
    The class allows to get the PDF of the estimated distribution, and samples can be generated
    """

    def __init__(self, consNN, MaxEntrLayer):
        """
        Parameters
        ----------
        consNN: tf.Model
            Tensorflow model estimating the constraint
        MaxEntrLayer: SimpleMultiMaxEntrLayer
            Layer designed to enforce maximum entropy formalism
        """

        super(SimpleMultiMaxEntrModel, self).__init__()
        self.consNN = consNN
        self.MELayer = MaxEntrLayer
        self.params = dict()                                                     # storage for global parameters in the sampling process

    def call(self, input, training=False):
        """
        PDF evaluation over the samples
        
        Parameters
        ----------
        input: array_like
            Samples (# of dims,# of data) to evaluate the PDF over
        training: bool, default: False
            Notify the neural network components if the result is used for training

        Returns
        -------
        tf.Tensor
            PDF of every input sample (# of data,)
        """
        return self.MELayer(
            self.consNN(input, training=training)
        )

    def train_step(self, inputs):
        """
        Logic of each iteration of the training loop
        
        Parameters
        ----------
        inputs: array_like
            Samples (# of data, # of dims) to identify the PDF over

        Returns
        -------
        Dict
            Dictionary containing the current evaluation of the loss
        """
        
        with tf.GradientTape() as tape:
            gx = self.consNN(inputs, training=True)
            lambda_ = self.MELayer.lambda_update(self.consNN(self.MELayer.intPts, training=True))
            gMean = tf.math.reduce_mean(gx)
            gamma = tf.math.add(
                    tf.math.multiply(self.MELayer.mu, gMean),
                    lambda_
            )
        grads = tape.gradient(gamma, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.MELayer.lambda_update(self.consNN(self.MELayer.intPts, training=True))

        return {"gamma": gamma}

 
    @tf.function
    def _generateProposal(self, data):
        """
        Proposes a new candidate set with one Metropolis-Hastings iteration

        Parameters
        ----------
        samp : array_like
            Current iteration of the sample (size, # of dims)

        Returns
        -------
        tf.Tensor
            New candidate set (size, # of dims)
        """
        shape = tf.shape(data)
        sigma = self.params["sig"]
        shift = tf.random.normal(shape, mean=0., stddev=sigma)
        proposal = data + shift

        probData = self(data)
        probProposal = self(proposal)

        ratios = tf.math.divide_no_nan(probProposal, probData)
        threshold = tf.random.uniform(tf.shape(ratios))
        maskAcceptance = tf.cast(tf.math.greater(ratios, threshold), dtype=tf.float32)[:, None]

        updatedData = proposal * maskAcceptance + data * (tf.constant(1.) - maskAcceptance) # check shape casting

        return updatedData, maskAcceptance

    @tf.function
    def _condSamplingLoop(self, i, samp):
        """
        Condition in the sampling loop, should only be used for it!

        Parameters
        ----------
        i : int
            Current iteration of the loop
        samp : array_like
            Current iteration of the sample (size, # of dims). Unused here!

        Returns
        -------
        bool
            Indicates if the number of burn in iteration has been fulfilled
        """

        burnIn = self.params["burnIn"]
        return i < burnIn
    
    @tf.function
    def _bodySamplingLoop(self, i, samp):
        """
        Condition in the sampling loop, should only be used for it!

        Parameters
        ----------
        i : int
            Current iteration of the loop
        samp : array_like
            Current iteration of the sample (size, # of dims)

        Returns
        -------
        int, tf.Tensor
            Iteration number incremented by one, newly proposed vector as a candidate sample
        """
        
        return [i+1, self._generateProposal(samp)[0]]

    @tf.function
    def rvs(self, size, initSample, burnIn=100, sig=1.):
        """
        Generate a sample with a method based on the Metropolis-Hastings method

        Instead of using a single chain and sampling correlated samples after the burn in, \"size\" chains are sampled in a vectorized independant manner. The first sample after the burn in is kept for every Markov Chain.

        Parameters
        ----------
        size : int
            Size of the required sample
        initSample : array_like
            Starting points (# of data, # of dims) of the Markov Chain
        burnIn : int, default: 100
            Burn in number of iterations. Can be relatively short if initSample is close to the distribution
        sig : float
            Standard deviation of the Gaussian noise used to propose candidate points

        Returns
        -------
        tf.Tensor
            A sample (size, # of dims) of the distribution           
        """

        multVal = tf.cast(tf.math.ceil(
            tf.math.divide(
                size,
                tf.shape(initSample)[0]
            )
        ), tf.int32)
        multTens = tf.stack([multVal, 1])
        startSample = tf.tile(tf.cast(initSample, tf.float32), multTens)[:size, :]
        noise = tf.random.normal(tf.shape(startSample), mean=0., stddev=.2*sig)
        startSample += noise

        self.params.update({"burnIn":burnIn, "size":size, "sig":sig})

        i0 = tf.constant(0)
        _, burntSample = tf.while_loop(self._condSamplingLoop, self._bodySamplingLoop, loop_vars=[i0, startSample], parallel_iterations=1)

        return burntSample