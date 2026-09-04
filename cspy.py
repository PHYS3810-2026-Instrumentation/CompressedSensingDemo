import os, sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, fft, fft2, ifft, ifft2
from scipy.linalg import dft
from scipy.signal import unit_impulse
from scipy.ndimage import zoom
import cvxpy as cp
from enum import Enum
from numba import jit

class Basis(Enum):
    RANDOM = 0
    DISCRETE_COSINE = 1
    FOURIER = 2

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
    if len(image.shape) == 3:
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
    else:
        x, y = image.shape
        if y > x:
            # crop along y (1) to make it only =x
            start = int((y/2) - (x/2))
            end = int((y/2) + (x/2))
            image = image[:,start:end]
        elif x > y:
            # crop along x (0) to make it only =y
            start = int((x/2) - (y/2))
            end = int((x/2) + (y/2))
            image = image[start:end,:]
        return image

def ltret(code, pipeline=False, payload = None):
    if not pipeline and payload == None:
        return code
    else:
        return (code, payload)

# @jit(nopython=False)
def _reconstruct(y, n, p, sparsity=5, epsilon=1e-10, max_iter=1000, mode='cvx', basis:Basis|any = Basis.DISCRETE_COSINE, verbose:bool = False):
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


def reconstruct(args, pipe:bool=False, inp:any = None):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy reconstruct


usage:
    py cspy.py reconstruct (path/to/vectors) 
    Reconstructs data based on the contents of the vectors. All vectors should be in the standard 1D sampled vector format. At least one vector file should be a sample position vector P.

arguments:
    -o, --out (path)
        specifies the save path for all of the output files. Xn.csv will be appended to each output.

    -b, --basis (basis)

''')
        return ltret(0, pipe)
    
# PIPE FORMAT: header, P, [Yn]
# INPUT PIPE: header, data
def sample(args, pipe:bool=False, inp:any = None):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy sample
Samples a given vectorized signal, saving under vectors Yn and P where Yn is the sampled data from channel 'n', and P is the sample position array. both .csv files will have the format of the source image (such as xy X Y), stored at the first line as a header comment (which can then be expressed to find the full size).

usage:
    py cspy.py sample (path/to/some/vector) xx%
    Takes a sample of xx% of the vector signal at the given path location.

arguments:
    -o, --out (path)
        specifies the save path for all of the output files. Yn.csv, P.csv will be appended per file. If not specified, the input path will be used.
''')
        return ltret(0, pipe)
    IN_PATH = None
    OUT_PATH = None
    PERCENT = None

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        else:
            if arg.endswith("%"):
                PERCENT = float(arg[:-1]) / 100.
            else:
                IN_PATH = arg

        idx += 1

    header, data = None, None
    if IN_PATH == None:
        if inp != None:
            IN_PATH = inp[0]
            header = inp[1][0]
            data = inp[1][1]
        else:
            print(f'No input given!')
            return ltret(-1, pipe)
    else:
        with open(IN_PATH, mode='r') as fp:
            header = fp.readline()
        data = np.loadtxt(IN_PATH, delimiter=',')

    header = header.strip('# ')
    if PERCENT == None:
        print(f'No percentage given!')
        return ltret(-2, pipe)

    if OUT_PATH == None:
        OUT_PATH = IN_PATH

    N = data.shape[0]
    channels = data.shape[1]

    P = take_samples(N, int(PERCENT * N))

    Y = []
    for channel in range(channels):
        temp = data[P, channel]
        Y.append(np.array(temp))

    np.savetxt(f'{OUT_PATH}.P.csv', P, fmt='%d', delimiter=',', header=header)
    for n in range(channels):
        np.savetxt(f'{OUT_PATH}.Y{n}.csv', Y[n], fmt='%f', delimiter=',\t', header=header)

    if pipe:
        return ltret(0, payload=(OUT_PATH, (header, P, Y)))
    
    return ltret(0, pipe)

# SEMICOMPLETE
# TODO implement other vectorize methods
# PIPE FORMAT: header, data
# INPUT PIPE: image
def vector(args, pipe:bool = False, inp:any = None):
    if type(args) == str:
        args = args.split()
    
    if len(args) == 0:
        print('''
    cspy vector
    Vectorizes a 2d width by height image into a 1D vector format. The method used to vectorize the 2D array is defined by the pattern argument, which defaults to 'xy'.
    
    usage:
        py cspy.py vector (path/to/some/image) 
        Vectorizes the image at the given path location.
    
    arguments:
        -o, --out (path)
            specifies the save path for all of the output files. If not specified, the input path will be used with X.csv appended.
        
        -p, --pattern (pattern)
            specifies the type of pattern to be used for vectorizing. valid patterns are 'xy', 'yx', 'zigzag', 'theta' and 'snake'. note that 'theta' and 'snake' will only work if the given image is square!
            defaults to 'xy'
        
        -r, --resize (w=(integer)|h=(integer))
            resizes the image before vectorizing such that either the X axis is the given width, or the Y axis is the given height
    
        -c, --crop
            takes the central crop of the image (prior to resizing), ensuring the image is square. 

        -n, --normalise
            normalises the image to a range (0, 1) if it is integer-typed (in a range 0-255).

        -s, --separate
            separates contents of a multi-channel image into individual channeled images. output filename should have a * to denote where the number should go if featured.
    ''')
        return ltret(0, pipe)
    IN_PATH = None
    OUT_PATH = None
    DO_RESIZE = None # can be a tuple with either first or second parameter being a positive integer to scale down to.
    DO_CROP = False
    PATTERN = 'xy'
    NORMALISE = False
    SEPARATE = False

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-p', '--pattern'):
            idx += 1
            PATTERN = args[idx]
        elif arg in ('-n', '--normalise'):
            NORMALISE = True
        elif arg in ('-s', '--separate'):
            SEPARATE = True
        elif arg in ('-r', '--resize'):
            idx += 1
            tmp = args[idx].split('=')
            if len(tmp) != 2:
                print(f'Wrong format of resize parameter. ({args[idx]})')
                return ltret(-1, pipe)
            elif tmp[0] not in ('w', 'h'):
                print(f'First part of resize parameter must be "w" or "h". ({args[idx]})')
                return ltret(-1, pipe)

            v = int(tmp[1])
            if tmp[0] == 'w':
                DO_RESIZE = (v, None)
            else:
                DO_RESIZE = (None, v)

        elif arg in ('-c', '--crop'):
            DO_CROP = True
        else:
            IN_PATH = arg

        idx += 1

    if IN_PATH == None:
        if inp != None:
            IN_PATH = inp[0]
            image = inp[1]
        else:
            print(f'No input given!')
            return ltret(-2, pipe)
    else:
        image = plt.imread(IN_PATH)

    MONO_IMAGE = len(image.shape) == 2
    if MONO_IMAGE:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    RAW_OUT_PATH = OUT_PATH
    if OUT_PATH == None:
        if SEPARATE:
            OUT_PATH = IN_PATH + '.X{0}.csv'
        else:
            OUT_PATH = f'{IN_PATH}.X.csv'
    elif SEPARATE:
        if '*' not in OUT_PATH:
            print(f'Out path does not have * character to denote index for separate channel files!')
            return ltret(-1, pipe)
        OUT_PATH = OUT_PATH.replace('*', '{0}')

    if DO_CROP:
        image = square_crop(image)

    if DO_RESIZE != None:
        factor = 1
        if DO_RESIZE[0] == None:
            # resize on H
            H = DO_RESIZE[1]
            h = image.shape[1]
            factor = H / h
        else:
            # resize on W
            W = DO_RESIZE[0]
            w = image.shape[1]
            factor = W / w

        image = zoom(image, (factor, factor, 1))

    if NORMALISE:
        ALLOWED_DATATYPES = [np.float32, np.float16, np.float64, np.floating]
        if type(min) not in ALLOWED_DATATYPES and type(max) not in ALLOWED_DATATYPES:
            image = image / 255.0

    VALID_PATTERNS = ['xy', 'yx', 'zigzag', 'theta', 'snake']
    if PATTERN is None:
        PATTERN = 'xy'

    if PATTERN not in VALID_PATTERNS:
        print(f'Pattern given {PATTERN} is not valid.')
        return ltret(-3, pipe)

    header = ''
    out = []
    match PATTERN:
        case 'xy':
            w, h, c = image.shape
            n = w*h
            out = np.reshape(image, (n, c))
            header = f'xy {w} {h}'
        case 'yx':
            w, h, c = image.shape
            image = np.transpose(image, axes=(1, 0, 2))
            n = w*h
            out = np.reshape(image, (n, c))
            header = f'yx {w} {h}'
        case 'zigzag':
            pass
        case 'theta':
            pass

    if not SEPARATE:
        np.savetxt(OUT_PATH, out, fmt='%f', delimiter=',\t', header=header)
        if pipe:
            return ltret(0, payload=(OUT_PATH, [f'# {header}', out]))
    else:
        n = 0
        columns = out.shape[1]
        while n < columns:
            data = out[:,n]
            np.savetxt(OUT_PATH.format(n), data, fmt='%f', delimiter=',', header=header)
            n += 1
        if pipe:
            return ltret(0, payload=(RAW_OUT_PATH, [f'# {header}', out]))
    
    return ltret(0, pipe)

# TODO implement the other vectorize methods, multiple data (up to 4 channels RGBA)
# PIPE FORMAT: image
# INPUT PIPE: header, data
# INPUT PIPE: header, [data]
def devector(args, pipe:bool = False, inp:any = None):
    if type(args) == str:
        args = args.split()
    
    if len(args) == 0:
        print('''
    cspy devector
    Devectorizes a signal back into an image.

    usage:
        py cspy.py devector (path/to/some/csv) 
        Devectorizes the signal stored in a csv file. The csv file should have the format stored as the header (i.e. xy {x} {y}).
        If the csv file path is in the form X*, searches for all numbered files X0, X1, ... and stitches all files together for an RGB/RGBA image.
    
    arguments:
        -o, --out (path)
            specifies the save path for the output image. Defaults to a .png rendition of the given input.
    ''')
        return ltret(0, pipe)

    IN_PATH = None
    OUT_PATH = None

    idx = 0
    while idx < len(args):
        arg = args[idx]

        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        else:
            IN_PATH = args[idx]
        idx += 1

    data = None
    header = None
    MULTI_DATA = False
    if IN_PATH == None:
        if inp != None:
            IN_PATH = inp[0]
            header = inp[1][0]
            data = inp[1][1]
            if type(data) == list:
                MULTI_DATA = True
        else:
            print(f'No file specified?')
            return ltret(-1, pipe)
    else:
        if '*' in IN_PATH:
            formatstr = IN_PATH.replace('*', '{0}')
            N = 0
            MULTI_DATA = True
            data = []
            while True:
                actualstr = formatstr.format(N)
                if not os.path.exists(actualstr):
                    break
                if N == 0:
                    with open(actualstr, mode='r') as fp:
                        header = fp.readline().strip('# ')
                temp = np.loadtxt(actualstr, delimiter=',')
                data.append(temp)
                N += 1
            if len(data) == 0:
                print(f'No files found by format {IN_PATH}')
                return ltret(-1, pipe)
            if len(data) > 4:
                print(f'Invalid number of channels {len(data)}')
                return ltret(-1, pipe)
            
            IN_PATH = IN_PATH.replace('*', 'n')
        else:
            with open(IN_PATH, mode='r') as fp:
                header = fp.readline().strip('# ')
            data = np.loadtxt(IN_PATH, delimiter=',')

    if OUT_PATH == None:
        OUT_PATH = f'{IN_PATH}.png'

    header = header.split()
    format_type = header[0]
    image = None
    match format_type:
        case 'xy':
            x = int(header[1])
            y = int(header[2])
            if not MULTI_DATA:
                c = data.shape[1]
                image = np.reshape(data, (x, y, c))
            else:
                image = np.zeros((x, y, len(data)))
                n = 0
                for vector in data:
                    channel = np.reshape(vector, (x, y))
                    image[:,:,n] = channel
                    n += 1
        case 'yx':
            x = int(header[1])
            y = int(header[2])
            if not MULTI_DATA:
                c = data.shape[1]
                image = np.reshape(data, (y, x, c))
                image = np.transpose(image, axes=(1, 0, 2))
            else:
                image = np.zeros(x, y, len(data))
                n = 0
                for vector in data:
                    channel = np.reshape(vector, (y, x, 1))
                    channel = np.transpose(channel, axes=(1, 0, 2))
                    image[:,:,n] = channel
                    n += 1


    plt.imsave(OUT_PATH, image)
    if pipe:
        return ltret(0, payload=(OUT_PATH, image))

    return ltret(0, pipe)

# PIPE FORMAT: image
# INPUT PIPE: image
def monochrome(args, pipe:bool = False, inp:any = None):
    if type(args) == str:
        args = args.split()
    
    if len(args) == 0:
        print('''
    cspy monochrome
    Turns an image into a monochrome image (RGB->G), discarding any alpha channels.

    usage:
        py cspy.py monochrome (path/to/some/image) 
    
    arguments:
        -o, --out (path)
            specifies the save path for the output image. Defaults to a suffixed .png rendition of the given input.
    ''')
        return ltret(0, pipe)

    IN_PATH = None
    OUT_PATH = None

    idx = 0
    while idx < len(args):
        arg = args[idx]

        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        else:
            IN_PATH = args[idx]
        idx += 1

    image = None
    if IN_PATH == None:
        if inp != None:
            IN_PATH = inp[0]
            image = inp[1]
        else:
            print(f'No input path given?')
            return ltret(-1, pipe)
    else:
        image = plt.imread(IN_PATH)

    if OUT_PATH == None:
        OUT_PATH = f'{IN_PATH}mono.jpg'

    if len(image.shape) == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        plt.imsave(OUT_PATH, image)

        if pipe:
            return ltret(0, payload=(OUT_PATH, image))

    ALLOWED_DATATYPES = [np.float32, np.float16, np.float64, np.floating]
    if type(min) not in ALLOWED_DATATYPES and type(max) not in ALLOWED_DATATYPES:
        image = image / 255.0

    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    outimg = np.sqrt(np.pow(r, 2) + np.pow(g, 2) + np.pow(b, 2))
    outimg = outimg * 255
    outimg = outimg.astype(int)

    plt.imsave(OUT_PATH, outimg, cmap='gray')

    if pipe:
        return ltret(0, payload=(OUT_PATH, outimg))
    return ltret(0, pipe)

def cspy_engine(args):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy
An implementation of a compressed sensing suite in Python.

usage:
    py cspy.py (suite|vector|sample|reconstruct|devector|monochrome|help) ...

suite:
    Runs a test suite compressed sensing reconstruction on an image. Allows selection of various sampling patterns, sampling percentages, image resizing, algorithms, and so on.
    Accepts images of arbitrary color channel count and arbitrary sizes, but can be resized down as specified. 
    If a reconstruction fails and logs were created, can be specified to resume from a given log path.

vector:
    Vectorizes a 2D image into a 1D signal. CSV signal files require the type of signal to be stored as a header comment.

sample:
    For a given image, generates a sample mask as specified and saves samples from all channels of the image (Yn) as well as the sample position array P.
    
reconstruct:
    Given appropriate data (samples and sample coordinates), reconstructs an image.
    Data should be given in two files, both containing 1-dimensional CSV formats - sample position array P, and sample array Y (where P stores the location of each Y in the full datastream of size N). 
    Basises can be chosen for reconstruction.

help:
    Displays information about command line parameters
''')
        exit(0)

    big_argstring = ' '.join(args)
    argstrings = big_argstring.split(',')
    if len(argstrings) > 1:
        prior_state = None
        print(f'Executing pipeline...')
        for args in argstrings:
            args = args.split()
            mode = args[0].lower()

            match mode:
                case 'sample':
                    code, prior_state = sample(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        exit(0)
                case 'vector':
                    code, prior_state = vector(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        exit(0)
                case 'devector':
                    code, prior_state = devector(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        exit(0)
                case 'reconstruct':
                    pass
                case 'help':
                    pass
                case 'monochrome':
                    code, prior_state = monochrome(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        exit(0)
                case _:
                    print(f'Unrecognized mode {mode}')
                    exit(-1)
    else:
        mode = args[0].lower()

        if mode == 'sample':
            exit(sample(args[1:]))
        elif mode == 'suite':
            pass
        elif mode == 'reconstruct':
            pass
        elif mode == 'help':
            pass
        elif mode == 'vector':
            exit(vector(args[1:]))
        elif mode == 'devector':
            exit(devector(args[1:]))
        elif mode == 'monochrome':
            exit(monochrome(args[1:]))
        else:
            print(f'Unrecognized mode {mode}')

if __name__ == '__main__':
    cspy_engine(sys.argv[1:])