import numpy as np


def rbfDot(X, Y, sigma):
    """
    Compute de radial basis function between every pair of vectors between two list
    
    Parameters
    ----------
    X, Y : array_like
        List of vectors (# of dim, # of data) to compute the rbf on
    sigma : float
        Bandwidth of the rbf
    
    Returns
    -------
    np.array
        Array (# of X data, # of Y data) of the rbf between the vector pairs
    """

    xx = np.square(X).sum(0, keepdims=True).T               # shape (# of X data, 1)
    yy = np.square(Y).sum(0, keepdims=True)                 # shape (1, # of Y data)
    xy = -2. * X.T @ Y                                      # shape (# of X data, # of Y data)

    return np.exp(
        (-.5 * sigma ** -2) * (xx + yy + xy)
    )


def sigmaCalc(X, Y, q=50.):
    """
    Compute the bandwidth for the RBF

    Parameters
    ----------
    X, Y : array_like
        List of vectors (# of dim, # of data) to determine sigma on
    q : float
        Percentile of the list of distances to use for sigma
    Returns
    -------
    float
        Value of the bandwidth
    """
    
    m = X.shape[1]
    V = np.concatenate((X, Y), axis=1)
    v2 = np.square(V).sum(0, keepdims=True)                     # shape (1, # of data)
    vv = -2. * V.T @ V
    dists = v2.T + v2 + vv                                      # the matrix dists is symmetric and the diagonal is filled with 0
    med = np.percentile(dists[np.tril_indices(2*m, -1)], q)     # only the values in the lower triangle are considered for finding the percentile value
    sigma = (.5 * med) **.5
    return sigma


def mmd2u(X, Y, sigma=-1.):
    """
    Evaluate the MMD squared unbiased between two samples
    
    Parameters
    ----------
    X, Y : array_like
        Samples of identical shapes (# of dim, # of data) to evaluate the MMD on
    sigma : float, default: -1.
        Bandwidth of the RBF used to computer the MMD. If left to a negative or null, the sigmaCalc function is called to choose it
    Returns
    -------
    float
        Value of the MMD2u
    """

    m = X.shape[1]
    if sigma <= 0.:
        sigma = sigmaCalc(X, Y)

    K = rbfDot(X, X, sigma)
    L = rbfDot(Y, Y, sigma)
    KL = rbfDot(X, Y, sigma)

    MMD2u = 1./(m * (m-1)) * np.sum((1. - np.eye(m)) * (K + L - KL - KL.T))
    return MMD2u


def mmdThresCalc(bsMMD, alpha):
    """
    Finds the threshold value of MMD to be under a level of type I error according to the null distribution
    
    Parameters
    ----------
    bsMMD : array_like
        Sorted array of the realisations of the MMD under the null distribution
    alpha : float
        Level of type I error wanted
    
    Returns
    -------
    float
        Maximal value of the MMD to be under the level of error sought
    """

    return bsMMD[np.round((1-alpha) * len(bsMMD)).astype(int)]


def mmd2uEigValSample(X, Y, sigma=-1., nbEig = -1, nbSamp=-1):
    """
    Generate a sorted sample of the null distribution of the MMD2u using the asymptotic distribution

    Parameters
    ----------
    X, Y : array_like
        List of vectors (# of dim, # of data) used to generate the asymptotic distribution
    sigma : float, default: -1.
        Bandwidth of the RBF. If negative or null, the function sigmaCalc is used to compute one
    nbEig : int, default: -1
        Number of eigenvalues used out of the Gram matrix. If negative or null, default to 2 * (# of data - 1)
    nbSamp : int
        Number of samples of the null distrubution generated. If negative or null, default to 2 * # of data
    
    Returns
    -------
    np.array
        Sample of realisation of the MMD2u under the null distribution
    """

    m = X.shape[1]

    if sigma <= 0.:
        sigma = sigmaCalc(X, Y)
        
    K = rbfDot(X, X, sigma)
    L = rbfDot(Y, Y, sigma)
    KL = rbfDot(X, Y, sigma)

    Kz = np.block([[K, KL], [KL.T, L]])                     # uncentered Gram matrix
    H = np.eye(2*m) - (1./(2.*m)) * np.ones((2*m, 2*m))     # centering matrix
    Kz = H @ Kz @ H                                         # centered Gram matrix

    if nbEig <= 0.:
        nbEig = 2*m - 2

    kEigs = np.linalg.eigvalsh(Kz)[2*m - nbEig:]            # eigenvalues are sorted in ascending order and only the biggest are used 
    kEigs = 1./(2.*m) * np.abs(kEigs)

    if nbSamp <= 0.:
        nbSamp = 2*m

    normSamp = np.square(2. * np.random.randn(nbEig, nbSamp)) - 4.
    nullSamp = (1./(2.*m)) * kEigs[None, :] @ normSamp

    return np.sort(nullSamp[0, :])


def pValueCalc(bsSamp, mmd):
    """
    Compute the p-value of a realisation of MMD compared to the null distribution
    
    Note: The p-value can not reach 0. If it is very small, a minimal value of 1/len(bsSamp) is assumed (resolution of the p-value).

    Parameters
    ----------
    bsSamp : array_like
        Sorted array of the realisations of the MMD under the null distribution
    mmd : float
        MMD value to compare with the null distribution
    
    Returns
    -------
    float
        p-value of the given mmd
    """

    L = len(bsSamp)
    ls = np.linspace(0., 1., L)
    ind = np.searchsorted(bsSamp, mmd)
    if ind == len(bsSamp):
        ind -= 1
    pVal = 1. - ls[ind]
    res = 1. / L
    decNum = np.floor(np.log10(L)).astype(int)
    pVal = min(round(pVal - .5 * res, decNum) + res, 1.)
    return pVal

if __name__ == "__main__":
    #Example with 2D Gaussian with a mean shift
    nb, dim, diff = 1000, 2, .3

    sample1 = np.random.randn(dim, nb)                                          # First sample
    sample2 = np.random.randn(dim, nb) + diff                                   # Second sample, shifted in mean

    mmdSample = mmd2u(sample1, sample2)
    print(mmdSample)

    mmdNullDistrSample = mmd2uEigValSample(sample1, sample2)

    thresVal = .05                                                              # Illustration with an usual value
    threshold = mmdThresCalc(mmdNullDistrSample, thresVal)
    print("threshold value at {:.1f}% is {}".format(thresVal * 100, threshold))
    pVal = pValueCalc(mmdNullDistrSample, mmdSample)
    print("p-value = {}".format(pVal))