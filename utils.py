"""
@author: Jaimy
"""
import numpy
import theano
import theano.tensor as T
import os


def NormalInit(rng, sizeX, sizeY, scale=0.01, sparsity=-1):
    """ 
    Normal Initialization
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)

    if sparsity < 0:
        sparsity = sizeY

    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    return values.astype(theano.config.floatX)


def OrthogonalInit(rng, shape):
    if len(shape) != 2:
        raise ValueError

    if shape[0] == shape[1]:
        # For square weight matrices we can simplify the logic
        # and be more exact:
        M = rng.randn(*shape).astype(theano.config.floatX)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        return Q

    M1 = rng.randn(shape[0], shape[0]).astype(theano.config.floatX)
    M2 = rng.randn(shape[1], shape[1]).astype(theano.config.floatX)

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = numpy.linalg.qr(M1)
    Q2, R2 = numpy.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * numpy.sign(numpy.diag(R1))
    Q2 = Q2 * numpy.sign(numpy.diag(R2))

    n_min = min(shape[0], shape[1])
    return numpy.dot(Q1[:, :n_min], Q2[:n_min, :])


def GrabProbs(class_probs, target, gRange=None):
    if class_probs.ndim > 2:
        class_probs = class_probs.reshape((class_probs.shape[0] * class_probs.shape[1], class_probs.shape[2]))
    else:
        class_probs = class_probs
    if target.ndim > 1:
        tflat = target.flatten()
    else:
        tflat = target

    # return grabber(class_probs, target).flatten()
    ### Hack for Theano, much faster than [x, y] indexing 
    ### avoids a copy onto the GPU
    return T.diag(class_probs.T[tflat])


def SoftMax(x):
    x = T.exp(x - T.max(x, axis=x.ndim - 1, keepdims=True))
    return x / T.sum(x, axis=x.ndim - 1, keepdims=True)


def save_df(output_dat_path, df, fn):
    file_path = output_dat_path + fn
    if not os.path.exists(output_dat_path):
        os.makedirs(output_dat_path)
    df.to_csv(file_path, index=False)
    print "{} saved".format(file_path)
