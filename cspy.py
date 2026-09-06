import os, sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, fft, fft2, ifft, ifft2
from scipy.linalg import dft
from scipy.signal import unit_impulse
from scipy.ndimage import zoom
import cvxpy as cp
from enum import Enum

class Basis(Enum):
    RANDOM = 0
    DISCRETE_COSINE = 1
    FOURIER = 2

_BASIS_FUNCTIONS = {
    Basis.RANDOM : (
        lambda p, n, y : (np.eye(n, n) - np.random.random((n, n)))[p,:],
        lambda p, n, theta, s : s,
    ),

    Basis.DISCRETE_COSINE : (
        lambda p, n, y : dct(np.eye(n, n))[p, :],
        lambda p, n, theta, s : idct(s),
    ),

    Basis.FOURIER : (
        lambda p, n, y : dft(n)[p,:],
        lambda p, n, theta, s : np.flip(ifft(s)),
    )
}


def cosamp(phi, u, s, epsilon=1e-10, max_iter=1000):
    """
    Return an `s`-sparse approximation of the target signal
    Input:
        - phi, sampling matrix
        - u, noisy sample vector
        - s, sparsity
    """
    a = np.zeros(phi.shape[1])
    v = u
    it = 0 # count
    halt = False
    while not halt:
        it += 1
        print("Iteration {}\r".format(it), end="")
        
        y = np.dot(np.transpose(phi), v)
        omega = np.argsort(y)[-(2*s):] # large components
        omega = np.union1d(omega, a.nonzero()[0]) # use set instead?
        phiT = phi[:, omega]
        b = np.zeros(phi.shape[1])
        # Solve Least Square
        b[omega], _, _, _ = np.linalg.lstsq(phiT, u)
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        v_old = v
        v = u - np.dot(phi, a)

        halt = (np.linalg.norm(v - v_old) < epsilon) or \
            np.linalg.norm(v) < epsilon or \
            it > max_iter
        
    return a

def take_samples(length, samples):
    '''
    returns a list of sampled indicies for X, such that x[indicies] returns only those samples
    '''
    selections = np.round(np.random.rand(samples, 1) * length-1)
    selections = [int(e[0]) for e in selections]
    return selections

def square_crop(image):
    x, y, _ = image.shape
    if y > x:
        # crop along y (1) to make it only =x
        start = int((y/2) - (x/2))
        end = int((y/2) + (x/2))
        image = image[:,start:end,:]
    elif x > y:
        # crop along x (0) to make it only =y
        start = int((x/2) - (y/2))
        end = int((x/2) + (y/2))
        image = image[start:end,:,:]
    return image

# @jit(nopython=False)
def reconstruct(y, n, p, sparsity=5, epsilon=1e-10, max_iter=1000, mode='cvx', basis:Basis|any = Basis.DISCRETE_COSINE, verbose:bool = False):
    '''
    for a list of samples y along with positions in the signal p (1:1), and length of output n, computes and returns the reconstructed signal x.

    the transformation basis can be set with the 'basis' argument, which should be a Basis enum set to a valid value. 

    a custom basis may be provided as a tuple of two functions - the first being a function of one parameter, n, that returns the transformation matrix theta, and the second being a function of two parameters, p & s (n being the length of s), that applies the inverse of the first function to produce x. 

    the default basis is Discrete Cosine transformation

    the algorithm mode can also be specified with the 'mode' argument. by default, the mode is l1 norm minimalisation via the CVXPY package. 
    '''
    if basis is None:
        basis = Basis.DISCRETE_COSINE

    generate = None
    inverse = None
    if type(basis) is Basis:
        generate, inverse = _BASIS_FUNCTIONS[basis]
    else:
        # TODO ensure check here to make sure no errors!
        generate, inverse = basis

    theta = generate(p, n, y)

    if theta is None:
        print(f'No theta could be found! theta is None')
        return None

    if mode=='cosamp':
        s = cosamp(theta, y, sparsity, epsilon, max_iter)
    elif mode=='cvx':
        s_l1 = cp.Variable(n, complex=True)
        obj = cp.Minimize(cp.norm(s_l1, 1))
        cons = [theta @ s_l1 == np.conjugate(y)]
        problem = cp.Problem(obj, cons)
        problem.solve(verbose=verbose)
        s = s_l1.value

    return inverse(p, n, theta, s)

def sample(args):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy sample
Samples a given image, saving under vectors Yn and P where Yn is the sampled data from channel 'n', and P is the sample position array. both .csv files will have the maximum size of the source image, N, stored as the first term.

usage:
    py cspy.py sample (path/to/some/image) xx%
    Takes a sample of xx% of the image at the given path location.

arguments:
    -o, --out (path)
        specifies the save path for all of the output files. Yn.csv, P.csv will be appended per file. If not specified, the input path will be used.
    
    -p, --pattern (pattern)
        specifies the type of pattern to be used for sampling. valid patterns are 'xy', 'yx', 'zigzag', and 'theta'. note that 'theta' will not work if the given image is square!
        defaults to 'xy'
    
    -r, --resize (w=(integer)|h=(integer))
        resizes the image before sampling such that either the X axis is the given width, or the Y axis is the given height

    -c, --crop
        takes the central crop of the image (prior to resizing), ensuring the image is square. 
''')
        return 0
    IN_PATH = None
    OUT_PATH = None
    DO_RESIZE = None # can be a tuple with either first or second parameter being a positive integer to scale down to.
    DO_CROP = False
    PERCENT = None
    PATTERN = 'xy'

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-p', '--pattern'):
            idx += 1
            PATTERN = args[idx]
        elif arg in ('-r', '--resize'):
            idx += 1
            tmp = args[idx].split('=')
            if len(tmp) != 2:
                print(f'Wrong format of resize parameter. ({args[idx]})')
                return -1
            elif tmp[0] not in ('w', 'h'):
                print(f'First part of resize parameter must be "w" or "h". ({args[idx]})')
                return -1

            v = int(tmp[1])
            if tmp[0] == 'w':
                DO_RESIZE = (v, None)
            else:
                DO_RESIZE = (None, v)
        elif arg in ('-c', '--crop'):
            DO_CROP = True
        else:
            if arg.endswith("%"):
                PERCENT = float(arg[:-1]) / 100.
            else:
                IN_PATH = arg

        idx += 1
    if IN_PATH == None:
        print(f'No input given!')
        return -2
    if PERCENT == None:
        print(f'No percentage given!')
        return -2

    if OUT_PATH == None:
        OUT_PATH = IN_PATH

    image = plt.imread(IN_PATH)

    if DO_CROP:
        image = square_crop(image)

    if DO_RESIZE != None:
        pass

    
    return 0


def main(args):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy
An implementation of a compressed sensing suite in Python.

usage:
    py cspy.py (suite|reconstruct|sample|help) ...

suite:
    Runs a test suite compressed sensing reconstruction on an image. Allows selection of various sampling patterns, sampling percentages, image resizing, algorithms, and so on.
    Accepts images of arbitrary color channel count and arbitrary sizes, but can be resized down as specified. 
    If a reconstruction fails and logs were created, can be specified to resume from a given log path.
    
reconstruct:
    Given appropriate data (samples and sample coordinates), reconstructs an image.
    Data should be given in two files, both containing 1-dimensional CSV formats - sample position array P, and sample array Y (where P stores the location of each Y in the full datastream of size N). Additionally requires the full length to be provided, N. Reconstruction will produce an array X and save it under a given path (optionally can be cast into a x by y image if specified). Multiple Y arrays can be specified.
    Basises can be chosen for reconstruction.

sample:
    For a given image, generates a sample mask as specified and saves samples from all channels of the image (Yn) as well as the sample position array P. Sample mask order and percentage can be determined.

help:
    Displays information about command line parameters
''')
        exit(0)
    mode = args[0].lower()

    if mode == 'sample':
        exit(sample(args[1:]))
    elif mode == 'suite':
        pass
    elif mode == 'reconstruct':
        pass
    elif mode == 'help':
        pass
    else:
        print(f'Unrecognized mode {mode}')



if __name__ == '__main__':
    main(sys.argv[1:])