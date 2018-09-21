
import numpy as np
from numpy import concatenate,matmul,mean
from numpy.linalg import solve
    
def approxProp(A, B, g, w):
    '''
    APPPROP an implementation of "all-pairs appearance-space edit propagation"
    im, input image in Lab color space and m-by-n-by-3 dimension
    g, a vector user specified edits' parameters
    w, a vector holds user specified strength of g, range in [0,1]
    imout, output image, edited using AppProp algorithm
    e, the propagated edit parameters
    '''
    U=concatenate((A,B))
    lamda=mean(w)

    one=matmul(0.5/lamda*U, solve(A,matmul(U.T,w)))
    two=matmul(U, solve(A, np.sum(U.T, axis=1, keepdims=True)))
    dinv=1/(one+two)

    itm1=dinv*matmul(U, solve(A,matmul(U.T,w*g)))
    itm2=dinv*matmul(U, solve(-A+matmul(U.T, dinv*U),matmul(U.T, itm1)))
    e=0.5/lamda*(itm1-itm2)

    return e

def fullProp(Z, g, w):
    lamda=mean(w)

    d=matmul(Z, 1+0.5*w/lamda)
    e=0.5/lamda*solve(np.diagflat(d)-Z, matmul(Z, w*g))

    return e
