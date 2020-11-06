import numpy as np
import numpy.random as npr
import numpy.linalg as la

# import autograd.numpy as np
# import autograd.numpy.random as npr
# import autograd.numpy.linalg as la


EPSILON = 1e-16

def count(self, axis=None):
    s = self.shape
    if axis is None:
        return self.size
    else:
        n = s[axis]
        t = list(s)
        del t[axis]
        return np.full(t, n, dtype=np.intp)


def matrix_power(a, n):

    if n == 1:
        return a

    elif n == 2:
        return np.matmul(a, a)

    elif n == 3:
        return np.matmul(np.matmul(a, a), a)

    result = a

    for i in range(n-1):
        result = np.matmul(result, a)

    return result
def moment(a, moment=1, axis=0):

    if moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return np.zeros(shape, dtype=float)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return np.float64(0.0)
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n-1)/2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        a_zero_mean = a - np.expand_dims(a.mean(axis), axis)
        if n_list[-1] == 1:
            s = a_zero_mean
        else:
            s = a_zero_mean**2

            # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)

#impliment skew using skew code from scipy for two dimensional
#array

def skew(a, axis=0, bias=True):
    n = a.shape[axis]
    m2 = moment(a, 2, axis)
    m3 = moment(a, 3, axis)
    olderr = np.seterr(all='ignore')
    try:
        vals = np.where(m2 == 0, 0, m3 / (m2**1.5+EPSILON))
    finally:
        np.seterr(**olderr)

    if not bias and n > 2:
            vals = np.where(m2 > 0,
                            np.sqrt((n-1.0)*n)/(n-2.0)*m3/(m2**1.5+EPSILON), vals)

    if vals.ndim == 0:
        return vals.item()

    return vals

#impliment kurtosis using the kurtosis code from scipy for two dimensional
#array

def kurtosis(a, axis=0, fisher=True, bias=True):

    #a = np.asanyarray(a)
    n = a.shape[axis]
    m2 = moment(a, 2, axis)
    m4 = moment(a, 4, axis)
    olderr = np.seterr(all='ignore')
    try:
        vals = np.where(m2 == 0, 0, m4 / (m2**2.0+EPSILON))
    finally:
        np.seterr(**olderr)

    if not bias and n > 3:

        vals = np.where(m2 > 0,
                        1.0/(n-2)/(n-3)*((n*n-1.0)*m4/(m2**2.0-3*(n-1)**2.0+EPSILON))+3.0,
                        vals)
    if vals.ndim == 0:
        return vals.item()

    return vals - 3 if fisher else vals
