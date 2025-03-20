"""
Custom distribution generators

Allows to regenerate and get the PDF of the distributions presented in the paper, not available in libraries
"""

import numpy as np
import scipy.stats as st
import sympy as sp

from scipy.special import roots_legendre


class SpikeGen(st.rv_continuous):
    """
    Spike distribution from Crespo_2018 (https://doi.org/10.1016/j.apm.2018.07.029)

    This class inherit from scipy.stats.rv_continuous, for more flexibility.
    
    After a probability vector transformation, the expression of the PDF is an integral with bounds depending on the variable X.
    It can be evaluated with a symbolic solver, but it takes a lot of time.
    The integral can be computed by the same solver, but the result is approximative and yield complex number (with very small imaginary component).
    It is however the prefered method as the approximation is good and very fast.
    """

    def __init__(self):
        super().__init__()

        self.u = st.uniform()
        self.v = st.uniform()

        self.x, y = sp.symbols("x y")
        lowBound = sp.Piecewise((sp.cbrt(-self.x), sp.StrictLessThan(self.x, 0)), (0, True))             # Lower bound of the integral
        highBound = sp.Piecewise((sp.cbrt(1-self.x), sp.StrictGreaterThan(self.x, 0)), (1, True))        # Higher bound of the integral
        px = sp.Integral(1 / ( 2 * sp.sqrt(self.x + y**3) ), (y, lowBound, highBound))                   # Body of the integral
        self.apx = px.doit()                                                                             # Letting the symbolic solver find an approximation

    def _rvs(self, size, **args):
        """
        st.rv_continuous method for sampling the distribution, should not be used directly
        
    Parameters
        ----------
        size: int
            Number of sample to generate
        args: Dict, optional
            Other parameters potentially used by scipy
            
        Returns
        -------
        np.array
            Array (size,) of the samples
        """

        return np.power(self.u.rvs(size=size, **args), 2) - np.power(self.v.rvs(size=size, **args), 3)
    
    def _pdf(self, x):
        """
        st.rv_continuous method for computing the PDF of the distribution, should not be used directly
        
        Parameters
        ----------
        x: array_like
            Samples (len(x),) to evaluate the PDF over
            
        Returns
        -------
        np.array
            Array (len(x),) of the density over the samples.
        """ 
       
        pts = []
        funcPdf = self.apx
        for el in x:
            mod = funcPdf.subs(self.x, el)
            eval = mod.evalf(chop=1e-6)                             # The PDF can only be evaluated sample by sample
            (pts.append(float(eval)))

        return np.abs(np.array(pts))                                # Transform the complex output (with a small imaginary part) to real


class CamelGen(st.rv_continuous):
    """
    Camel distribution with 2 truncated Gaussian distribution
    
    This class inherit from scipy.stats.rv_continuous, for more flexibility.
    """

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14):
        """
        Parameters
        ----------
        momtype : int, optional
            The type of generic moment calculation to use: 0 for pdf, 1 (default) for ppf.
        a : float, optional
            Lower bound of the support of the distribution, default is minus infinity.
        b : float, optional
            Upper bound of the support of the distribution, default is plus infinity.
        xtol : float, optional
            The tolerance for fixed point calculation for generic ppf.
        """

        super().__init__(momtype, a, b, xtol)

        clip_g1 = -5., 3.
        clip_g2 = -3., 5.
        loc1, scale1 = -1., .7
        loc2, scale2 = 1., .7
        lim_g1 = (clip_g1[0] - loc1) / scale1, (clip_g1[1] - loc1) / scale1 
        lim_g2 = (clip_g2[0] - loc2) / scale2, (clip_g2[1] - loc2) / scale2 

        self.g1 = st.truncnorm(*lim_g1, loc=loc1, scale = scale1)
        self.g2 = st.truncnorm(*lim_g2, loc=loc2, scale = scale2)

    def _pdf(self, x):
        """
        st.rv_continuous method for computing the PDF of the distribution, should not be used directly
        
        Parameters
        ----------
        x: array_like
            Samples (len(x),) to evaluate the PDF over
            
        Returns
        -------
        np.array
            Array (len(x),) of the density over the samples.
        """ 

        return .5 * (self.g1.pdf(x) + self.g2.pdf(x))
    
    def _cdf(self, x):
        """
        st.rv_continuous method for computing the CDF of the distribution, should not be used directly
        
        Parameters
        ----------
        x: array_like
            Samples (len(x),) to evaluate the CDF over
            
        Returns
        -------
        np.array
            Array (len(x),) of the CDF over the samples.
        """ 

        return .5 * (self.g1.cdf(x) + self.g2.cdf(x))
    
    def _rvs(self, size, **args):
        """
        st.rv_continuous method for sampling the distribution, should not be used directly
        
        Parameters
        ----------
        size: int
            Number of sample to generate
        args: Dict, optional
            Not used, but necessary for scipy
            
        Returns
        -------
        np.array
            Array (size,) of the samples
        """
        
        r1 = self.g1.rvs(size=size)
        r2 = self.g2.rvs(size=size)
        ber = st.bernoulli.rvs(.5, size=size)
        return r1 * ber + r2 * (1 - ber)
    

def spiralPDF(V, sigma=.5, r_s=1.5, L=np.pi, nBranches=3, startAngle=0., nInt=35):
    """
    PDF function of the spiral distribution

    In the dataset of Wenliang19a (https://proceedings.mlr.press/v97/wenliang19a.html / https://github.com/kevin-w-li/deep-kexpfam), the PDF of the Spiral distribution is not provided.
    This uses an analytical expression of the PDF (with the "eps" value kept to 1).
    The analytical expression is an integral, and the integration is handled by the Gauss-Legendre method.

    Parameters
    ----------
    V : array_like
        Array of coordinates (dim, nb) over which evaluate the PDF
    sigma : float, default: 0.5
        Standard deviation of the Gaussian noise used
    r_s : float, default: 1.5
        Radius of the spiral
    L : float, default: np.pi
        Angle between the start and the end of the branches
    nBranches : int, default: 3
        Number of branches of the Spiral
    startAngle : float, default: 0.
        Rotate all the spiral by the desired angle
    nInt : int, default: 35
        Number of integration points used for the Gauss-Lengendre method

    Returns
    -------
    np.array
        Array (nb,) of the density evaluated over the input array
    """

    V = V.T
    a, b, c = r_s*L, L, L*sigma/r_s                                                                                   # parameters of the PDF

    angle = 2./nBranches*np.pi
    cosSin = [(np.cos(i * angle + startAngle), np.sin(i * angle + startAngle)) for i in range(nBranches)]             # Cosine and Sine for every branch 

    # Only the PDF of a single branch is known, implying the PDF must be evaluated for every rotation
    X = np.concatenate([co * V[:, 0:1] + si * V[:, 1:2] for co, si in cosSin]).T
    Y = np.concatenate([co * V[:, 1:2] - si * V[:, 0:1] for co, si in cosSin]).T

    roots, weights = roots_legendre(nInt)
    zMat = .5 * (roots[:, None] + 1.)

    denomVec = c * zMat + .1
    t1Vec = 1. / (2. * np.pi * np.square(denomVec))
    t2Vec = (X - a * zMat * np.cos(b * zMat)) / denomVec
    t3Vec = (Y - a * zMat * np.sin(b * zMat)) / denomVec

    pBranch_xyz = t1Vec * np.exp(-.5 * (np.square(t2Vec)+ np.square(t3Vec)))
    pBranch_xy = .5 * weights @ pBranch_xyz
    pSpiral = pBranch_xy.reshape(nBranches, V.shape[0]).mean(0)                                                       # The evalutions at every rotation are averaged

    return pSpiral


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    grid = np.stack(np.meshgrid(np.linspace(-6., 6., 100), np.linspace(-6., 6, 100)), axis=-1).reshape((100**2,2))

    pdf = spiralPDF(grid.T, startAngle=1./4. * np.pi)

    img = np.reshape(pdf, (100, 100))

    plt.figure()
    plt.imshow(img, origin='lower')
    plt.show()