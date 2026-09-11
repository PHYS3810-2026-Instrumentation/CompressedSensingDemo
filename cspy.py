import os, sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, fft, fft2, ifft, ifft2
from scipy.linalg import dft, hadamard
from scipy.signal import unit_impulse
from scipy.ndimage import zoom
import cvxpy as cp
from enum import Enum
from numba import jit
# import pywt
from sklearn.decomposition import PCA

'''
Functions necessary to enable various transform basises
'''

def _get_haar_slices(n):
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'haar', level=max_level, mode='symmetric')
    _, slices = pywt.coeffs_to_array(coeffs)
    return slices

def _get_db_slices(n):
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'db4', level=max_level, mode='symmetric')
    _, slices = pywt.coeffs_to_array(coeffs)
    return slices

def _get_bo_slices(n):
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'bior4.4', level=max_level, mode='symmetric')
    _, slices = pywt.coeffs_to_array(coeffs)
    return slices

def _get_cdf53_slices(n):
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'bior2.2', level=max_level, mode='symmetric')
    _, slices = pywt.coeffs_to_array(coeffs)
    return slices

def _get_cdf971_slices(n):
    """
    Dynamically tracks asymmetric coefficient layouts and calculates the 
    exact total shape requirements for CDF 9/7 sub-bands.
    """
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'bior6.8', level=max_level, mode='symmetric')
    
    # coeff_arr contains the exact total length of all asymmetric sub-bands combined
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)
    return slices, coeff_arr.size

def _boundary_safe_cdf971_reconstruct(s, n):
    """
    Reconstructs the image patch using shape-safe arrays to accommodate 
    the jagged sub-band dimensions native to CDF 9/7.
    """
    slices, expected_size = _get_cdf971_slices(n)
    s_flat = np.real(s).flatten()
    
    # Create a destination array matching the exact total shape needed by the slices
    padded_s = np.zeros(expected_size, dtype=float)
    
    # Copy the solved sparse coefficients into the reconstruction canvas safely
    num_to_copy = min(s_flat.size, expected_size)
    padded_s[:num_to_copy] = s_flat[:num_to_copy]
        
    # Unflatten the safely sized jagged coefficients array back into PyWavelets structure
    coeffs = pywt.array_to_coeffs(padded_s, slices, output_format='wavedec')
    
    # Reconstruct and slice back down to the target block dimensions
    recon = pywt.waverec(coeffs, 'bior6.8', mode='symmetric')
    return recon[:n]

def _get_cdf972_slices(n):
    """
    Dynamically tracks the asymmetric coefficient shapes and total flat size 
    requirements for the bior3.5 (CDF 9/7 variant) transform.
    """
    max_level = pywt.dwt_max_level(data_len=n, filter_len=9)
    coeffs = pywt.wavedec(np.zeros(n), 'bior3.5', level=max_level, mode='symmetric')
    
    # coeff_arr contains the exact total length of all asymmetric sub-bands combined
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)
    return slices, coeff_arr.size

def _boundary_safe_cdf972_reconstruct(s, n):
    """
    Safely pads partial sparse tracking arrays to accommodate the 
    asymmetric sub-band structures of the bior3.5 transform.
    """
    slices, expected_size = _get_cdf972_slices(n)
    s_flat = np.real(s).flatten()
    
    # Pre-allocate a zero canvas matching the exact required coefficient array size
    padded_s = np.zeros(expected_size, dtype=float)
    
    # Safely copy sparse coefficients without out-of-bounds errors
    num_to_copy = min(s_flat.size, expected_size)
    padded_s[:num_to_copy] = s_flat[:num_to_copy]
        
    # Unflatten back into PyWavelets coefficient lists
    coeffs = pywt.array_to_coeffs(padded_s, slices, output_format='wavedec')
    
    # Reconstruct and slice back down to the target block dimensions
    recon = pywt.waverec(coeffs, 'bior3.5', mode='symmetric')
    return recon[:n]

def _get_sequency_ordered_hadamard(n):
    """
    Generates a perfectly orthogonal, sequency-ordered (Walsh) Hadamard matrix.
    This packs the maximum energy into the lowest indices.
    """
    H = hadamard(n)
    # Count the zero-crossings (sequency) for each row
    sign_changes = np.sum(H[:, :-1] * H[:, 1:] < 0, axis=1)
    # Sort matrix rows from lowest sequency (DC trend) to highest (Fine details/noise)
    sort_indices = np.argsort(sign_changes)
    return H[sort_indices, :] / np.sqrt(n)

def _hadamard_forward_with_edge_handling(p, n):
    """
    Generates the row projection matrix. 
    Using a normalized sequency matrix natively stabilizes basis transitions.
    """
    H_walsh = _get_sequency_ordered_hadamard(n)
    return H_walsh[p, :]

def _hadamard_inverse_with_edge_handling(s, n):
    """
    Reconstructs the spatial domain signal and applies a windowed edge filter 
    or basic clipping to clean up boundary noise artifacts.
    """
    H_walsh = _get_sequency_ordered_hadamard(n)
    # Project back to spatial domain. H_walsh is symmetric and orthogonal -> H_inv = H.T
    recon = np.dot(H_walsh.T, s)
    
    # Boundary post-processing: If your upstream pipeline runs edge mirroring, 
    # you would trim it here. For a standalone block, we stabilize the edge values:
    if len(recon) > 4:
        recon[0] = (recon[0] + recon[1]) / 2
        recon[-1] = (recon[-1] + recon[-2]) / 2
        
    return recon

def _fourier_forward_smooth_edges(p, n):
    """
    Generates a complex-valued Fourier basis row as a 2D row vector (1, n).
    Preserving complex values ensures the CVX solver converges without generating NaNs.
    """
    identity_row = np.zeros(n)
    identity_row[p] = 1.0
    
    fourier_row = dct(identity_row, norm='ortho')
    
    return fourier_row[np.newaxis, :]

def _fourier_inverse_smooth_edges(s, n):
    """
    Safely pads partial subsampled coefficients to length 'n' before running 
    the inverse transform, eliminating out-of-bounds index crashes.
    """
    s_flat = np.asarray(s).flatten()
    
    # Pad s_flat with zeros up to full patch length 'n' if it is a subsampled array
    if s_flat.size < n:
        padded_s = np.zeros(n, dtype=complex)
        padded_s[:s_flat.size] = s_flat
    elif s_flat.size > n:
        padded_s = s_flat[:n]
    else:
        padded_s = s_flat
    
    # Reverse the Fourier mapping
    recon = idct(padded_s, norm='ortho')
    
    # Edge-blending filter to smooth out blocking artifacts
    if n > 4:
        recon = (recon + recon) / 2
        recon[-1] = (recon[-1] + recon[-2]) / 2
        
    return recon

def _native_morlet2(M, s, w=5.0):
    """
    Pure NumPy implementation of the Complex Morlet Wavelet.
    Replaces the removed scipy.signal.morlet2 function.
    """
    # Create a time vector centered at zero
    x = np.arange(0, M) - (M - 1) / 2.0
    
    # Scale calculation: x / s
    x_scaled = x / s
    
    # Mathematical Morlet equation: complex exponential * gaussian envelope
    wavelet = np.exp(1j * w * x_scaled) * np.exp(-0.5 * x_scaled**2)
    
    # Normalise the wavelet based on standard L2 energy normalization
    wavelet *= np.pi**(-0.25) * np.sqrt(1.0 / s)
    return wavelet

def _get_morlet_basis_matrix(n, omega=5.0):
    """
    Generates a full 2D Morlet wavelet basis matrix of size n x n.
    Rows represent shifted/scaled Morlet wavelets with edge-tapering.
    """
    matrix = np.zeros((n, n), dtype=complex)
    
    # Generate multi-scale rows (mixing low-frequency and high-frequency scales)
    # Lower rows handle smooth broad trends; higher rows handle sharp transients
    scales = np.linspace(1.5, n / 4, n)
    
    for p in range(n):
        s = scales[p]
        # Generate the complex Morlet wavelet centered in the patch chunk
        wavelet = _native_morlet2(n, s, w=omega)
        
        # Shift the wavelet peak relative to row index 'p' for spatial coverage
        shift = int(p - n // 2)
        wavelet = np.roll(wavelet, shift)
        
        # Apply a smooth window to handle boundary reflections gracefully
        window = np.hanning(n)
        matrix[p, :] = wavelet * window
        
    # Standardize and normalize matrix rows to prevent scaling imbalances
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return matrix / row_norms

def _morlet_forward(p, n):
    """Generates a 2D complex Morlet basis row matrix entry."""
    M = _get_morlet_basis_matrix(n)
    return np.array([M[p, :].flatten()])

def _morlet_inverse(s, n):
    """
    Performs stable pseudo-inverse reconstruction to reconstruct the 
    spatial domain image signal from complex coefficients.
    """
    M = _get_morlet_basis_matrix(n)
    s_flat = np.asarray(s).flatten()
    
    # Zero-pad partial subsampled arrays to maintain structural match shapes
    if s_flat.size < n:
        padded_s = np.zeros(n, dtype=complex)
        padded_s[:s_flat.size] = s_flat
    elif s_flat.size > n:
        padded_s = s_flat[:n]
    else:
        padded_s = s_flat
        
    # Because Morlet is not strictly orthogonal, we use the Moore-Penrose 
    # pseudo-inverse (pinv) to safely reverse the tracking matrix mapping
    M_inv = np.linalg.pinv(M)
    recon = np.real(np.dot(M_inv, padded_s))
    
    # Stabilize boundaries to suppress blocking artifacts
    if n > 4:
        recon = (recon + recon) / 2
        recon[-1] = (recon[-1] + recon[-2]) / 2
        
    return recon

'''
Data structures, global variables & useful functions
'''

VALID_PATTERNS = ['xy', 'yx', 'zigzag', 'theta', 'snake_square', 'snake_square_inv']

_TLC = np.sqrt(2) 
_TLC_SM = 0.95

BASIS_ALIASES = {
        'dct' : 'discrete-cosine',
        'fft' : 'fourier-transform',
        'fourier' : 'fourier-transform',
        'ha' : 'haar',
        'hdd' : 'hadamard',
        'db' : 'daubechies',
        'kl' : 'karhunen-loeve',
        'bo' : 'biorthagonal',
        'cdf53' : 'cohen-daubechies-feauveau-5-3',
        'cdf971' : 'cohen-daubechies-feauveau-1st-9-7',
        'cdf972' : 'cohen-daubechies-feauveau-2nd-9-7',
        'ihdd' : 'improved-hadamard',
        'ifft' : 'improved-fourier',
        'ml' : 'morlet',
}

ACCEPTED_BASISES = [
        'discrete-cosine',
        'fourier-transform',
        'hadamard',
        'haar',
        'daubechies',
        'karhunen-loeve',
        'biorthagonal',
        'cohen-daubechies-feauveau-5-3',
        'cohen-daubechies-feauveau-1st-9-7',
        'cohen-daubechies-feauveau-2nd-9-7',
        'improved-hadamard',
        'improved-fourier',
        'morlet',
]


KL_basis_matrix = PCA(n_components=1024).fit(np.eye(1024)).components_

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

def csv_header(path):
    header, data = None, None
    with open(path, mode='r') as fp:
        header = fp.readline()
    data = np.loadtxt(path, delimiter=',')
    header = header.strip('#').strip()
    return header, data

def multi_csv_header(path):
    path = path.replace('*', '{0}')
    n = 0
    header = None
    data = []
    while True:
        pathactual = path.format(n)
        if not os.path.exists(pathactual):
            break
        if header == None:
            with open(pathactual, mode='r') as fp:
                header = fp.readline()
        tmp = np.loadtxt(pathactual, delimiter=',')
        data.append(tmp)
        n += 1
    return n, header, data

def infer_length(header):
    header = header.split()
    fmat = header[0]
    w, h = int(header[1]), int(header[2])
    if fmat != 'theta':
        return w*h
    else:
        R = (w/2.) * _TLC
        return int(np.pi * np.pow(R, 2) / _TLC_SM)

def image_shape_props(image):
    if len(image.shape) == 3:
        return image.shape
    return image.shape[0], image.shape[1], 1

def is_intlike_image(image:np.ndarray)->bool:
    '''
    Returns true if the image is an int-like image (i.e. valued from 0-255) but is not stored with the int type. False otherwise.'''
    mx, mn = np.max(image), np.min(image)
    _FLOAT_LIKE = (np.float32, np.float64)
    if image.dtype not in _FLOAT_LIKE:
        return False
    if mx > 1 or mn > 1:
        return True
    return False

def remove_filetype_path(path):
    prefix = ''
    if path.startswith('./'):
        path = path[2:]
        prefix = './'
    tmp = path.split('.')
    if len(tmp) <= 1:
        return prefix + path
    return prefix + '.'.join(tmp[:-1])

def get_filetype_path(path):
    prefix = ''
    if path.startswith('./'):
        path = path[2:]
        prefix = './'
    tmp = path.split('.')
    if len(tmp) <= 1:
        return ''
    return tmp[-1]

# create theta conversion grids (xy->t) and (t->xy)
def theta_convert(radius):
    # 2 pi r steps (rounded) per radius, radius * 1.5 steps outwards
    # so effectively pi (r*1.5)^2
    # call it pi r*1.6^2, and only take the N steps

    width = int(2*radius)
    grid = np.zeros((width, width))
    l = int(np.pi * np.pow(radius * _TLC, 2) / _TLC_SM)
    squid = np.zeros((l, 2))

    max_R = int(radius * _TLC)
    N = 0
    for r in np.arange(0, max_R):
        max_Th = int(r * np.pi * 2 * (1/_TLC_SM))
        for theta in np.linspace(0, 2*np.pi, max_Th):
            X = np.round((np.cos(theta) * r) + radius).astype(int)
            Y = np.round((np.sin(theta) * r) + radius).astype(int)
            if X in range(0, width) and Y in range(0, width):
                grid[Y, X] = N
                squid[N] = (X, Y)
            N += 1
    squid = np.array(squid)
    squid = squid.astype(int)
    grid = grid.astype(int)
    return grid, squid

# create outer snake square path grids (xy->t) and (t->xy)
def snake_square_path(w, h):
    N = w * h
    grid = np.zeros((w, h), dtype=int)
    squid = np.zeros((N, 2))

    n = 0
    precess = [w, h]
    presign = [+1, -1]
    flip = 0
    squid[-1] = [0, 0]
    while n < N:
        # generate a trace path and an N path
        stride = precess[flip]
        walk = presign[flip] * stride
        n_stride = np.arange(n, n+stride)
        rootx, rooty = squid[n-1]
        # print(f'({rootx}, {rooty}) ({stride}), flip={flip}, walk={walk}')
        if flip == 0:
            # width stride
            walk_set = np.arange(rootx, rootx+walk, step=presign[flip], dtype=int)
            stand_set = np.array([rooty] * stride, dtype=int)
            grid[stand_set, walk_set] = n_stride
            squid[n_stride, 0] = walk_set
            squid[n_stride, 1] = stand_set
        else:
            # height stride
            walk_set = np.arange(rooty, rooty+walk, step=presign[flip], dtype=int)
            stand_set = np.array([rootx] * stride, dtype=int)
            grid[walk_set, stand_set] = n_stride
            squid[n_stride, 0] = stand_set
            squid[n_stride, 1] = walk_set

        n += stride

        # flip flip and decrement precess
        flip = 1-flip
        precess[flip] = precess[flip] - 1
        presign[flip] = -presign[flip]

    squid = np.array(squid)
    squid = squid.astype(int)
    grid = grid.astype(int)
    return grid, squid

# creates an inverted version of the snake_square_path
def snake_inv_square_path(w, h):
    N = w * h
    grid, squid = snake_square_path(w, h)
    grid = N - grid
    squid = np.flip(squid, axis=0)
    return grid, squid

# TODO
def RECONSTRUCT_DRIVER(Y, P, N, basis, verbose:bool = False):
    '''
    Runs a reconstructive algorithm for sampled data Y with positions P, and signal length N, with a given basis defined by the basis attribute. Returns the reconstructed signal.
    '''

    # MODIFICATION:
    # Add the basis name to ACCEPTED_BASISES. Can optionally add aliases (shortcuts) in the BASIS_ALIASES dict.
    # (Both of these structures are above)
    # Then add a case to the match statement below for your basis string and implement it there.

    if basis in BASIS_ALIASES:
        basis = BASIS_ALIASES[basis]

    if basis not in ACCEPTED_BASISES:
        return None

    match basis:
        case 'discrete-cosine':
            theta = dct(np.eye(N, N))[P,:]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return np.real(idct(s_l1.value))
        case 'fourier-transform':
            theta = dft(N)[P,:]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return np.real(np.flip(ifft(s_l1.value)))
        case 'hadamard':
            theta = (hadamard(N) / np.sqrt(N))[P, :]
            
            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return np.dot(hadamard(N) / np.sqrt(N), s_l1.value)

        case 'haar':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row,'haar', level=3, mode='symmetric'))[0] for row in np.eye(N,N)])[P,:],

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1.value), _get_haar_slices(N), output_format='wavedec'),'haar', mode = 'symmetric')

        case 'daubechies':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'db4', mode='symmetric'))[0]for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1), _get_db_slices(N), output_format='wavedec'), 'db4', mode='symmetric')

        case 'karhunen-loeve':
            theta = KL_basis_matrix[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return np.dot(PCA(n_components=N).fit(np.eye(N)).components_.T, s_l1)

        case 'biorthagonal':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior4.4', level=3, mode='symmetric'))[0]for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1), _get_bo_slices(N), output_format='wavedec'), 'bior4.4', mode='symmetric')

        case 'cohen-daubechies-feauveau-5-3':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior2.2', mode='symmetric'))[0]for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1), _get_cdf53_slices(N), output_format='wavedec'), 'bior2.2', mode='symmetric')

        case 'cohen-daubechies-feauveau-1st-9-7':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior6.8', level=pywt.dwt_max_level(data_len=N, filter_len=9), mode='symmetric'))[0] for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _boundary_safe_cdf971_reconstruct(s_l1, N)

        case 'cohen-daubechies-feauveau-2nd-9-7':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior3.5', level=pywt.dwt_max_level(data_len=N, filter_len=9), mode='symmetric'))[0] for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _boundary_safe_cdf972_reconstruct(s_l1, N)

        case 'improved-hadamard':
            theta = _hadamard_forward_with_edge_handling(P,N)

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _hadamard_inverse_with_edge_handling(s_l1, N)

        case 'morlet':
            theta = _morlet_forward(P,N)

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _morlet_inverse(s_l1,N)
    return np.zeros(N)

'''
Command-line parsing functions
'''

# PIPE FORMAT: header, [Xn]
# INPUT PIPE: header, P, [Yn]
def reconstruct(args, pipe:bool=False, inp:any = None):
    if type(args) == str:
        args = args.split()

    if len(args) == 0 and inp == None:
        basises = '\n'.join([f'\t\t{e} ({', '.join([k[0] for k in BASIS_ALIASES.items() if k[1] == e])})' for e in ACCEPTED_BASISES])
        print(f'''
cspy reconstruct

usage:
    py cspy.py reconstruct (path/to/vectors) 
    Reconstructs data based on the contents of the vectors. All vectors should be in the standard 1D sampled vector format. At least one vector file should be a sample position vector 'P.csv' - all others would be of form 'Yn.csv', but the names can be arbitrary.

arguments:
    -o, --out (path)
        specifies the save path for all of the output files. Xn.csv will be appended to each output.
    
    -v, --verbose
        toggles on verbosity of the CVX solver

    -b, --basis (basis)
        specifies the transformation basis to be used. The default basis is 'discrete-cosine'.
        Supported transformation bases (aliases):
{basises}

''')
        return ltret(0, pipe)

    BASIS = 'discrete-cosine'
    IN_PATH = []
    OUT_PATH = None
    VERBOSE = False

    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-b', '--basis'):
            idx += 1
            BASIS = args[idx]
        elif arg in ('-v', '--verbose'):
            VERBOSE = True
        else:
            IN_PATH.append(arg)
        idx += 1

    # process all file paths
    P = None
    Y = []
    header = None
    P_path = None

    if len(IN_PATH) == 0:
        if inp != None:
            IN_PATH = [inp[0]]
            pack = inp[1]
            P_path = IN_PATH[0]
            header = pack[0]
            P = pack[1]
            Y = pack[2]
        else:
            print(f'No input paths given!')
            return ltret(-1, pipe)
    else:
        for path in IN_PATH:
            if '*' in path:
                N, _, tmp = multi_csv_header(path)
                Y += tmp
            else:
                tH, tD = csv_header(path)
                if 'P.csv' in path:
                    P_path = path
                    P = tD
                    header = tH
                else:
                    Y.append(tD)

    if P is None:
        print(f'No P vector given!')
        return ltret(-1, pipe)

    if len(Y) == 0:
        print(f'No sample vectors given!')
        return ltret(-2, pipe)

    if OUT_PATH == None:
        # print(P_path)
        OUT_PATH = remove_filetype_path(P_path) + 're_'

    N = infer_length(header)
    # We now have P, Y and N. we now can do our processes!
    X = []
    if type(P) is not list:
        P = P.astype(int)
    n = 0
    for Yn in Y:
        tmp = RECONSTRUCT_DRIVER(Yn, P, N, BASIS, verbose=VERBOSE)
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        # print(f'{np.min(tmp)} -> {np.max(tmp)}')
        if tmp is None:
            print(f'CVX didn\'t work! Likely a bad basis specification...')
            return ltret(-2, pipe)
        X.append(tmp)
        np.savetxt(f'{OUT_PATH}X{n}.csv', tmp, fmt='%f', delimiter=',', header=header)
        n += 1

    if pipe:
        return ltret(0, payload=(OUT_PATH, [header, X]))

    return ltret(0, pipe)

    
# PIPE FORMAT: header, P, [Yn]
# INPUT PIPE: header, data
# INPUT PIPE: image
def sample(args, pipe:bool=False, inp:any = None):
    if type(args) == str:
        args = args.split()

    if len(args) == 0 and inp == None:
        print('''
cspy sample
Samples a given vectorized signal, saving under vectors Yn and P where Yn is the sampled data from channel 'n', and P is the sample position array. both .csv files will have the format of the source image (such as xy X Y), stored at the first line as a header comment (which can then be expressed to find the full size).

usage:
    py cspy.py sample (path/to/some/vector) xx%
    Takes a sample of xx% of the vector signal at the given path location.

arguments:
    -o, --out (path)
        specifies the save path for all of the output files. Yn.csv, P.csv will be appended per file. If not specified, the input path will be used.

    -m, --mask (path)
        specifies the path to a sample mask from which the samples will be taken.
        sample mask should be an all-or-none vector of the same length as the input vector
''')
        return ltret(0, pipe)
    IN_PATH = None
    OUT_PATH = None
    PERCENT = None
    MASK_PATH = None
    IS_IMAGE_SAMPLE = False

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-m', '--mask'):
            idx += 1
            MASK_PATH = args[idx]
        else:
            if arg.endswith("%"):
                PERCENT = float(arg[:-1]) / 100.
            else:
                IN_PATH = arg

        idx += 1

    header, data = None, None
    IMAGE_TYPES = ('jpg', 'jpeg', 'png')
    if IN_PATH == None:
        if inp != None:
            IN_PATH = inp[0]
            inp_payl = inp[1]
            if IN_PATH.split('.')[-1] in IMAGE_TYPES:
                data = inp_payl
                header = None
                IS_IMAGE_SAMPLE = True
            else:
                header = inp[1][0]
                data = inp[1][1]
                IS_IMAGE_SAMPLE = False
                header = header.strip('#').strip()
        else:
            print(f'No input given!')
            return ltret(-1, pipe)
    else:
        if IN_PATH.split('.')[-1] in IMAGE_TYPES:
            data = plt.imread(IN_PATH)
            header = None
            IS_IMAGE_SAMPLE = True
        else:
            header, data = csv_header(IN_PATH)
            IS_IMAGE_SAMPLE = False

    if OUT_PATH == None:
        OUT_PATH = remove_filetype_path(IN_PATH) + '_'
        if IS_IMAGE_SAMPLE:
            OUT_PATH += 'sample.jpg'

    if IS_IMAGE_SAMPLE:
        w, h = data.shape[0], data.shape[1]
        N = w*h
    else:
        N = data.shape[0]
        channels = data.shape[1]
    
    P = None
    if MASK_PATH != None:
        m_header, m_data = csv_header(MASK_PATH)
        # ensure both headers match?
        if header != None and m_header != header:
            pass
        if m_data.shape[0] != N:
            print(f'Invalid sample vector size: {m_data.shape[0]} mismatch {N}')
            return ltret(-10, pipe)
        # turn it into a 1d if it isnt already (i.e. 1 where any in the row is 1)

        m_data = np.where(m_data != 0, True, False)
        if len(m_data.shape) > 1:
            if m_data.shape[1] == 4:
                m_data = m_data[:,0:3]
            m_data = np.any(m_data, axis=(1))
        P = np.argwhere(m_data)[:,0]
        P = np.reshape(P, (P.shape[0]))
    else:
        if PERCENT == None:
            print(f'No percentage given!')
            return ltret(-2, pipe)
        P = take_samples(N, int(PERCENT * N))

    if IS_IMAGE_SAMPLE:
        sample_mask = np.zeros((N, 3))
        sample_mask[P] = np.ones(3) * 255
        sample_mask = sample_mask.astype(np.uint8)
        w, h = data.shape[0], data.shape[1]
        sample_mask_image = np.reshape(sample_mask, (w, h, 3))
        plt.imsave(OUT_PATH, sample_mask_image)
        if pipe:
            return ltret(0, payload=(OUT_PATH, sample_mask_image))
    else:
        Y = []
        for channel in range(channels):
            temp = data[P, channel]
            Y.append(np.array(temp))

        np.savetxt(f'{OUT_PATH}P.csv', P, fmt='%d', delimiter=',', header=header)
        for n in range(channels):
            np.savetxt(f'{OUT_PATH}Y{n}.csv', Y[n], fmt='%f', delimiter=',\t', header=header)

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
    
    if len(args) == 0 and inp == None:
        print(f'''
    cspy vector
    Vectorizes a 2d width by height image into a 1D vector format. The method used to vectorize the 2D array is defined by the pattern argument, which defaults to 'xy'.
    
    usage:
        py cspy.py vector (path/to/some/image) 
        Vectorizes the image at the given path location.
    
    arguments:
        -o, --out (path)
            specifies the save path for all of the output files. If not specified, the input path will be used with X.csv appended.
        
        -p, --pattern (pattern)
            specifies the type of pattern to be used for vectorizing. valid patterns are shown below. note that 'theta' will only work if the given image is square!
            defaults to 'xy'
            patterns: 
                {VALID_PATTERNS}
        
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
            OUT_PATH = remove_filetype_path(IN_PATH) + '_X{0}.csv'
        else:
            OUT_PATH = remove_filetype_path(IN_PATH) + '_X.csv'
        RAW_OUT_PATH = OUT_PATH.format('*')
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
            w, h, c = image.shape
            if w != h:
                print(f'Error! Theta operation requires square image. dimensions {w}, {h}')
                return ltret(-5, pipe)
            radius = w/2.0
            grid, squid = theta_convert(radius)

            x, y = squid[:,0], squid[:, 1]
            out = image[y, x]
            header = f'theta {w} {h}'
        case 'snake_square':
            w, h, c = image_shape_props(image)

            grid, squid = snake_square_path(w, h)
            x, y = squid[:,0], squid[:, 1]
            out = image[y, x]
            header = f'snake_square {w} {h}'
        case 'snake_square_inv':
            w, h, c = image_shape_props(image)

            grid, squid = snake_inv_square_path(w, h)
            x, y = squid[:,0], squid[:, 1]
            out = image[y, x]
            header = f'snake_square_inv {w} {h}'
            

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
    
    if len(args) == 0 and inp == None:
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
            header = inp[1][0].strip('# ')
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
        OUT_PATH = remove_filetype_path(IN_PATH) + '_devector.png'

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
        case 'theta':
            x = int(header[1])
            y = int(header[2])
            rad = x/2.
            grid, squid = theta_convert(rad)

            x_addr, y_addr = squid[:,0], squid[:, 1]
            if MULTI_DATA:
                image = np.zeros((x, y, len(data)))
                for n in range(len(data)):
                    image[y_addr, x_addr, n] = data[n]
            else:
                if len(data.shape) == 2:
                    c = data.shape[1]
                else:
                    c = 1
                image = np.zeros((x, y, c))
                image[y_addr, x_addr] = data
        case 'snake_square':
            w = int(header[1])
            h = int(header[2])

            grid, squid = snake_square_path(w, h)
            x_addr, y_addr = squid[:,0], squid[:, 1]
            if MULTI_DATA:
                image = np.zeros((w, h, len(data)))
                for n in range(len(data)):
                    image[y_addr, x_addr, n] = data[n]
            else:
                if len(data.shape) == 2:
                    c = data.shape[1]
                else:
                    c = 1
                image = np.zeros((w, h, c))
                image[y_addr, x_addr] = data
        case 'snake_square_inv':
            w = int(header[1])
            h = int(header[2])

            grid, squid = snake_inv_square_path(w, h)
            x_addr, y_addr = squid[:,0], squid[:, 1]
            if MULTI_DATA:
                image = np.zeros((w, h, len(data)))
                for n in range(len(data)):
                    image[y_addr, x_addr, n] = data[n]
            else:
                if len(data.shape) == 2:
                    c = data.shape[1]
                else:
                    c = 1
                image = np.zeros((w, h, c))
                image[y_addr, x_addr] = data

    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:,:,0]

    if is_intlike_image(image):
        image = image.astype(np.uint8)
    plt.imsave(OUT_PATH, image)
    if pipe:
        return ltret(0, payload=(OUT_PATH, image))

    return ltret(0, pipe)

# PIPE FORMAT: image
# INPUT PIPE: image
def monochrome(args, pipe:bool = False, inp:any = None):
    if type(args) == str:
        args = args.split()
    
    if len(args) == 0 and inp == None:
        print('''
    cspy monochrome
    Turns an image into a monochrome image (RGB->G), discarding any alpha channels.

    usage:
        py cspy.py monochrome (path/to/some/image) 
    
    arguments:
        -o, --out (path)
            specifies the save path for the output image. Defaults to a suffixed '_mono' rendition of the given input.
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
        OUT_PATH = remove_filetype_path(IN_PATH) + '_mono.' + get_filetype_path(IN_PATH)

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

def cspy_engine(args, pipe:bool=False, inp:any = None, print_cli:bool = False):
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
    Data should be given in two paths, both containing 1-dimensional CSV formats - sample position array P, and sample array Y (where P stores the location of each Y in the full datastream of size N).

devector:
    Converts CSV vector files back into image files.

monochrome:
    Converts a RGB image into a monochromatic (grayscale) image.
    
''')
        return

    big_argstring = ' '.join(args)
    argstrings = big_argstring.split(',')
    if len(argstrings) > 1:
        prior_state = inp
        if print_cli:
            print(f'Executing pipeline...')

        for args in argstrings:
            args = args.split()
            mode = args[0].lower()

            match mode:
                case 'sample':
                    code, prior_state = sample(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        return code
                case 'vector':
                    code, prior_state = vector(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        return code
                case 'devector':
                    code, prior_state = devector(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        return code
                case 'reconstruct':
                    code, prior_state = reconstruct(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        return code
                case 'help':
                    pass
                case 'monochrome':
                    code, prior_state = monochrome(args[1:], pipe=True, inp=prior_state)
                    if code != 0:
                        return code
                case _:
                    print(f'Unrecognized mode {mode}')
                    return -1
        if pipe:
            return (0, prior_state)
        return 0
    else:
        mode = args[0].lower()

        if mode == 'sample':
            return sample(args[1:], pipe, inp)
        elif mode == 'suite':
            pass
        elif mode == 'reconstruct':
            return reconstruct(args[1:], pipe, inp)
        elif mode == 'vector':
            return vector(args[1:], pipe, inp)
        elif mode == 'devector':
            return devector(args[1:], pipe, inp)
        elif mode == 'monochrome':
            return monochrome(args[1:], pipe, inp)
        else:
            if print_cli:
                print(f'Unrecognized mode {mode}')
            return ltret(-1, pipe)

if __name__ == '__main__':
    exit(cspy_engine(sys.argv[1:]))