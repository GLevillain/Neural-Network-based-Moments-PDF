"""
Monovariate Maximum Entropy Neural Network Probability Density Function model

This file provides the necessary elements to replicate the monovariate tests of the paper.
"""

import numpy as np
import tensorflow as tf


def trapezoidInt(n, min_, max_):
    """
    Provides points and associated weights for trapeze integration
    
    Parameters
    ----------
    n : int
        Number of points to use
    min_, max_ : float
        Lower and upper bounds of the integration
    
    Returns
    -------
    array_like of float, array_like of float
        The array of the integration points (n,), and the array of the associated weights (n,)
    """

    x = np.linspace(min_, max_, n)
    w = np.full_like(x, (max_-min_) / (n-1.))             # the expresssion of the trapeze integration is a telescopic sum
    w[[0, -1]] = .5 * w[[0, -1]]                          # all the weights are the same except the ends of the array
    return x.astype(np.float32), w.astype(np.float32)


class SimpleMaxEntrLayer(tf.keras.layers.Layer):
    """
    Layer enforcing the maximum entropy distribution formalism
    """
    
    def __init__(self, intPts, intWeights=None):
        """
        Parameters
        ----------
        intPts : array_like
            Points (# of data,) used to evaluate the integral
        intWeights : array_like, default: None
            Associated weights (# of data,) of the points, if none are provided unifom weights are used               
        """

        super(SimpleMaxEntrLayer, self).__init__()
        self.mu = self.add_weight(shape=(), trainable=True, name="mu")
        self.lambda_ = self.add_weight(shape=(), trainable=False, name="lambda")

        if intWeights is None:
            nbPts = intPts.shape[0]
            intWeights = np.full_like(intPts, 1./nbPts)
        
        self.logWeights = tf.constant(
            tf.math.log(intWeights[:, None]),
            dtype="float32",      
            name='log weights'
            )
        self.intPts = tf.constant(
            intPts[:, None],
            dtype="float32",           
            name="integration points"
            )

    def call(self, inputs):
        """
        Defines the outputs when the layer is called
        
        Parameters
        ----------
        inputs: array_like
            Scalar outputs of evaluations (# of data,) of the neural network constraint estimation
        
        Returns
        -------
        tf.Tensor
            The estimated PDF values (# of data,) for the given constraint evaluations
        """

        exponents = tf.math.add(
            tf.math.scalar_mul(self.mu, inputs),
            self.lambda_
        )
        return tf.math.exp(-exponents)[:, 0]
  
    def lambdaUpdate(self, intConsOut):
        """
        Update the integration constant of the estimated PDF
        
        Parameters
        ----------
        intConsOut: array_like
            Evaluations (# of data,) of the constraint for the integration set

        Returns
        -------
        tf.Tensor
            Updated value of lambda ()
        """

        lambda_ = tf.math.reduce_logsumexp(
            tf.math.subtract(
                    self.logWeights,
                    tf.math.scalar_mul(self.mu, intConsOut)
                )
            )
        self.lambda_.assign(lambda_)
        return lambda_



class SimpleMaxEntrModel(tf.keras.Model):
    """
    Model enforcing the maximum entropy distribution formalism
    
    The class allows to get the PDF, CDF & PPF of the estimated distribution, and samples can be generated
    """

    def __init__(self, consNN, MaxEntrLayer, intDelta=0.05):
        """
        Parameters
        ----------
        consNN: tf.Model
            Tensorflow model estimating the constraint
        MaxEntrLayer: SimpleMaxEntrLayer
            Layer designed to enforce maximum entropy formalism
        intDelta: float, default: 0.05
            Resolution of the domain discretization for trapezoid integration
            """
        
        super(SimpleMaxEntrModel, self).__init__()
        self.consNN = consNN
        self.MELayer = MaxEntrLayer

        self.intDelta = intDelta   

    def call(self, input, training=False):
        """
        PDF evaluation over the samples
        
        Parameters
        ----------
        input: array_like
            Samples (# of data,) to evaluate the PDF over
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
            Samples (# of data,) to identify the PDF over

        Returns
        -------
        Dict
            Dictionary containing the current evaluation of the loss
        """

        with tf.GradientTape() as tape:
            gx = self.consNN(inputs, training=True)
            # lambda_ = self.MELayer.lambdaUpdate(self.consNN(tf.stop_gradient(self.MELayer.intPts, training=True)))  # Only mu is tracked
            lambda_ = self.MELayer.lambdaUpdate(self.consNN(self.MELayer.intPts, training=True))  # Everything is tracked
            gMean = tf.math.reduce_mean(gx)
            gamma = tf.math.add(
                    tf.math.multiply(self.MELayer.mu, gMean),
                    lambda_
            )
        grads = tape.gradient(gamma, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.MELayer.lambdaUpdate(self.consNN(self.MELayer.intPts, training=True))
        return {"gamma": gamma}

    def cdf(self, inputs):
        """
        CDF evaluation over the samples

        Parameters
        ----------
        input: array_like
            Samples (# of data,) to evaluate the CDF over

        Returns
        -------
        tf.Tensor
            CDF (# of data,) of every input sample
        """

        #Sort the inputs values and their indexes, and also compute the reverse index operation.
        indSorted = tf.argsort(inputs)
        indInvSorted = tf.argsort(indSorted)
        inpSorted = tf.gather(inputs, indSorted)

        #Set the lower integration bound, and evenly sample the integration domain
        minVal = self.MELayer.intPts[0, 0]
        range_ = tf.range(minVal, inpSorted[-1], self.intDelta)

        #Get the density of the inputs and over the integration domain
        inpSortedDensity = self(inpSorted, training=False)
        rangeDensity = self(range_, training=False)

        #Search where the inputs would end up in a mixed array with the integration domain, and calculate the length of said domain
        inpInsertInd = tf.searchsorted(range_, inpSorted) + tf.range(0, tf.shape(inputs)[0])    # The i-th element is shifted by i in the concatenated sorted array 
        arrLen = tf.shape(inputs)[0] + tf.shape(range_)[0]

        #Insert the inputs values and density in the mixed array
        scatInp = tf.scatter_nd(inpInsertInd[:, None], inpSorted, [arrLen])
        scatInpDensity = tf.scatter_nd(inpInsertInd[:, None], inpSortedDensity, [arrLen])

        #Find where the sampled domain points would end up in the mixed array
        maskRange = tf.math.logical_not(tf.scatter_nd(inpInsertInd[:, None], tf.ones_like(inpSorted, dtype=tf.bool), [arrLen]))
        rangeInsertInd = tf.boolean_mask(tf.range(0, arrLen), maskRange)

        #Insert the sampled domain points in the mixed array
        scatRange = tf.scatter_nd(rangeInsertInd[:, None], range_, [arrLen])
        scatRangeDensity = tf.scatter_nd(rangeInsertInd[:, None], rangeDensity, [arrLen])

        #Assemble the mixed arrays
        randVar = scatInp + scatRange
        densities = scatInpDensity + scatRangeDensity

        #Calculate the integral by trapezoid integration
        trapezoidArea = tf.math.multiply(
            tf.math.scalar_mul(.5, tf.cast(densities[1:] + densities[:-1], tf.float64)), 
            randVar[1:] - randVar[:-1]
        )
        allIntegrals = tf.pad(tf.math.cumsum(trapezoidArea)[None, :], tf.constant([[0, 0], [1, 0]]), "CONSTANT")[0, :]

        #Extract from all the integral values those from the inputs
        inpSortedIntegral = tf.gather(allIntegrals, inpInsertInd)
        inpIntegral = tf.gather(inpSortedIntegral, indInvSorted)

        return inpIntegral
    
    @tf.function
    def _ppfIter(self, xOld, prob):
        """
        Iteration of the modified Newton-Raphson algorithm

        This method is designed to be used only by the ppf method loop.
        The modification handles the points choosen outside the domain of the distribution.
        Instead of going out of the domain, the proposed points sits in the middle of the original point and the breached bound.

        Parameters
        ----------
        xOld: tf.Tensor
            Previous candidate set of points (# of data,)
        prob: array_like
            Probabilites to compute the PPF on (# of data,)
        
        Returns
        -------
        tf.Tensor
            New set of candidate points (# of data,)
        """

        min_ = tf.cast(self.MELayer.intPts[0, 0], tf.float64)
        max_ = tf.cast(self.MELayer.intPts[-1, 0], tf.float64)
        Fx = tf.cast(self.cdf(xOld), tf.float64)
        dFdx = tf.cast(self.call(xOld), tf.float64)

        xNew = tf.math.subtract(                                           # Newton-Raphson iteration
            xOld,
            tf.math.divide_no_nan(
                tf.math.subtract(Fx, prob),
                dFdx
            )
        )
        mask = tf.math.logical_or(                                         # Detecting points out of bounds
            tf.math.greater_equal(xNew, max_),
            tf.math.less_equal(xNew, min_)
        )
        mask = tf.cast(mask, tf.float64)
        xNew =  tf.clip_by_value(xNew, min_, max_)                         # Clipping the points out of bounds

        return (
            (tf.constant(1., dtype=tf.float64) - mask) * xNew              # Update the candidate points in the range
            + mask * tf.constant(.5, dtype=tf.float64) * (xNew + xOld)     # Candidate points outside the range are updated by dichotomy
        )

    def ppf(self, prob, iter=10):
        """
        PPF evaluation over probabilities
        
        This method uses the CDF method with multiple iteration of the Newton-Raphson method to estimate the PPF at given probabilities
        
        Parameters
        ----------
        prob: array_like
            Probabilities (# of data,) to compute the PPF on
        iter: int, default: 10
            Number of iteration of the Newton-Raphson method
        
        Returns
        -------
        tf.Tensor
            Associated coordinates (# of data,) to the given probabilities
        """

        midRange = tf.cast(.5 * (self.MELayer.intPts[-1, 0] - self.MELayer.intPts[0, 0]), tf.float64)
        x = tf.math.scalar_mul(midRange, tf.ones_like(prob, dtype=tf.float64))

        for _ in range(iter):
            x = self._ppfIter(x, prob)
        return x
    
    def rvs(self, size):
        """
        Samples the distribution
        
        Parameters
        ----------
        size: int
            Number of samples desired
            
        Returns
        -------
        tf.Tensor
            A sample of the distribution (size,)
        """
        
        rand = tf.random.uniform((size,), dtype=tf.float64)
        sample = self.ppf(rand)
        return sample