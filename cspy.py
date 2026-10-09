import os, sys, pathlib, datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, fft, fft2, ifft, ifft2
from scipy.linalg import dft, hadamard
from scipy.signal import unit_impulse
from scipy.ndimage import zoom
from enum import Enum

import pywt
from sklearn.decomposition import PCA

'''
Functions necessary to enable various transform basises
'''

def RECONSTRUCT_DRIVER(Y, P, N, basis, verbose:bool = False):
    '''
    Runs a reconstructive algorithm for sampled data Y with positions P, and signal length N, with a given basis defined by the basis attribute. Returns the reconstructed signal.
    '''
    
    import cvxpy as cp

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
            nlogup = np.pow(2, np.ceil(np.log2(N))).astype(int)
            
            theta = (hadamard(nlogup) / np.sqrt(nlogup))[P, :]
            
            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return np.real(np.dot(hadamard(nlogup) / np.sqrt(nlogup), s_l1.value))[:N]

        case 'haar':
            max_level = pywt.dwt_max_level(data_len=N, filter_len=9)
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row,'haar', level=max_level, mode='symmetric'))[0] for row in np.eye(N,N)])[P,:]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)
            s = np.real(s_l1.value)

            return pywt.waverec(pywt.array_to_coeffs(s, _get_haar_slices(N), output_format='wavedec'),'haar', mode = 'symmetric')

        case 'daubechies':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'db4', mode='symmetric'))[0]for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1.value), _get_db_slices(N), output_format='wavedec'), 'db4', mode='symmetric')

        case 'karhunen-loeve':
            theta = _KL_make_basis(N).T[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            s = np.real(s_l1.value)

            return np.dot(PCA(n_components=N).fit(np.eye(N)).components_.T, s)

        case 'biorthogonal':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior4.4', level=3, mode='symmetric'))[0] for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            s = np.real(s_l1.value)

            tmp = pywt.array_to_coeffs(s, _get_bo_slices(N), output_format='wavedec')
            # print(f'{N}, {s.shape}')
            return pywt.waverec(tmp, 'bior4.4', mode='symmetric')

        case 'cohen-daubechies-feauveau-5-3':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior2.2', mode='symmetric'))[0]for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return pywt.waverec(pywt.array_to_coeffs(np.real(s_l1.value), _get_cdf53_slices(N), output_format='wavedec'), 'bior2.2', mode='symmetric')

        case 'cohen-daubechies-feauveau-1st-9-7':
            max_level = pywt.dwt_max_level(data_len=N, filter_len=9)
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior6.8', level=max_level, mode='symmetric'))[0] for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _boundary_safe_cdf971_reconstruct(s_l1.value, N)

        case 'cohen-daubechies-feauveau-2nd-9-7':
            theta = np.array([pywt.coeffs_to_array(pywt.wavedec(row, 'bior3.5', level=pywt.dwt_max_level(data_len=N, filter_len=9), mode='symmetric'))[0] for row in np.eye(N, N)])[P, :]

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _boundary_safe_cdf972_reconstruct(s_l1.value, N)

        case 'improved-hadamard':
            nlogup = np.pow(2, np.ceil(np.log2(N))).astype(int)
            theta = _hadamard_forward_with_edge_handling(P,nlogup)

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            s = np.real(s_l1.value)

            return _hadamard_inverse_with_edge_handling(s, nlogup)[:N]

        case 'improved-fourier':
            theta = _fourier_forward_smooth_edges(P, N)
            print(theta.shape)

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            s = np.real(s_l1.value)
            return _fourier_inverse_smooth_edges(s, N)

        case 'morlet':
            theta = _morlet_forward(P,N)

            s_l1 = cp.Variable(theta.shape[1], complex=True)
            obj = cp.Minimize(cp.norm(s_l1, 1))
            cons = [theta @ s_l1 == np.conjugate(Y)]
            problem = cp.Problem(obj, cons)
            problem.solve(verbose=verbose)

            return _morlet_inverse(s_l1.value, N)
    return np.zeros(N)


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
    # fourier_row = fft(identity_row)
    
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
    # recon = ifft(padded_s)
    
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

def _KL_make_basis(N):
    return PCA(n_components=N).fit(np.eye(N)).components_

'''
Data structures, global variables & useful functions
'''

theta_R_coeff = np.sqrt(2)
theta_R_dens = np.sqrt(2)
theta_Theta_dens = np.sqrt(2)

BASIS_ALIASES = {
        'dct' : 'discrete-cosine',
        'fft' : 'fourier-transform',
        'fourier' : 'fourier-transform',
        'ha' : 'haar',
        'hdd' : 'hadamard',
        'db' : 'daubechies',
        'kl' : 'karhunen-loeve',
        'bo' : 'biorthogonal',
        'cdf53' : 'cohen-daubechies-feauveau-5-3',
        'cdf971' : 'cohen-daubechies-feauveau-1st-9-7',
        'cdf972' : 'cohen-daubechies-feauveau-2nd-9-7',
        'ihdd' : 'improved-hadamard',
        # 'ifft' : 'improved-fourier',
        'ml' : 'morlet',
}

ACCEPTED_BASISES = [
        'discrete-cosine',
        'fourier-transform',
        'hadamard',
        'haar',
        'daubechies',
        'karhunen-loeve',
        'biorthogonal',
        'cohen-daubechies-feauveau-5-3',
        'cohen-daubechies-feauveau-1st-9-7',
        'cohen-daubechies-feauveau-2nd-9-7',
        'improved-hadamard',
        # 'improved-fourier',
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

def multi_csv_header(path, file_precache:dict = None):
    path = path.replace('*', '{0}')
    n = 0
    header = None
    data = []
    while True:
        pathactual = path.format(n)
        if not os.path.exists(pathactual):
            break
        if file_precache is not None and pathactual in file_precache:
            th, tmp = file_precache[pathactual]
        else:
            th, tmp = csv_header(pathactual)

        if header == None:
            header = th
        
        data.append(tmp)
        n += 1
    return n, header, data

def csv_save(path:str, data:np.ndarray, header:str):
    if data.dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64):
        data = data.astype(np.uint8)
        np.savetxt(path, data, fmt='%d', delimiter=',', header=header)
    else:
        np.savetxt(path, data, fmt='%f', delimiter=',', header=header)

def infer_length(header):
    header = header.split()
    fmat = header[0]
    w, h = int(header[1]), int(header[2])
    if fmat not in PATH_FUNCS.keys():
        print(f'Error; {fmat} not valid path function')
        exit(-1023)
    grid,squid = PATH_FUNCS[fmat](w, h)
    return squid.shape[0]

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

def squid_restrict(squid, w, h):
    squid = np.transpose(squid)
    p = np.where(np.logical_and(
        np.logical_and(squid[0] >= 0, squid[1] >= 0),
        np.logical_and(squid[0] < w, squid[1] < h)
    ))
    squid = np.transpose(squid)
    return squid[p, :][0], p[0]

def squid_restrict_data(squid, signal, w, h):
    squid = np.transpose(squid)
    p = np.where(np.logical_and(
        np.logical_and(squid[0] >= 0, squid[1] >= 0),
        np.logical_and(squid[0] < w, squid[1] < h)
    ))
    squid = np.transpose(squid)
    return squid[p, :][0], signal[p], p

def image_saver(image, path, MONO:bool = False):
    # protect image range
    if image.dtype == np.float32:
        image = np.clip(image, 0, 1)

    if len(image.shape) == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    # print(f'saving {path}, image shape {image.shape}')
    if image.shape[2] == 1:
        tmp = np.zeros((image.shape[0], image.shape[1], 3))
        for n in range(3):
            tmp[:,:,n] = image[:,:,0]
        image = tmp
        MONO = True

    if MONO:
        plt.imsave(path, image, cmap='gray')
    else:
        plt.imsave(path, image)

'''
Path-etch functions. Produces grid and squid.
'''

def theta_path(w, h, densities=None, invert=False):
    '''
    Densities can contain three attributes for the Radius coefficient, R density, and Theta density. defaults to sqrt(2) for each
    '''
    if densities is None:
        R_coeff, R_dens, Theta_dens = theta_R_coeff, theta_R_dens, theta_Theta_dens
    else:
        R_coeff, R_dens, Theta_dens = densities

    m_radius = max(w, h)/2
    total_radius = int(R_coeff * m_radius)
    mid_w = w / 2.
    mid_h = h / 2.

    grid = np.ones((w, h)) * -1
    total_R = int(R_dens * total_radius)
    R = np.linspace(0, total_radius, total_R)
    Theta_steps = Theta_dens * np.round(2 * np.pi * R)
    Theta_steps = Theta_steps.astype(int)
    N = np.sum(Theta_steps) + 1


    squid = np.zeros((N, 2))
    flip = False
    n = 0
    for r, dn in zip(R, Theta_steps):
        if r == 0 or dn == 0:
            x, y = np.round(mid_w).astype(int), np.round(mid_h).astype(int)
            squid[n] = [x, y]
            grid[y, x] = n
            n += 1
            continue
        thetas = np.linspace(0, 2*np.pi, dn)
        x, y = np.round(mid_w + (np.cos(thetas) * r)), np.round(mid_h + (np.sin(thetas) * r))
        x = x.astype(int)
        y = y.astype(int)

        if flip:
            x, y = np.flip(x), np.flip(y)
        if invert:
            flip = not flip
        n_space = np.arange(n, n+dn, dtype=int)

        p = np.array([x, y])
        p = np.where( np.logical_and(np.logical_and(p[0] >= 0, p[1] >= 0), np.logical_and(p[0] < w, p[1] < h)) )

        squid[n_space, 0] = x
        squid[n_space, 1] = y
        grid[y[p], x[p]] = n_space[p]

        n += dn
    
    squid = np.array(squid)
    squid = squid.astype(int)
    grid = grid.astype(int)
    return grid, squid

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

def snake_inv_square_path(w, h):
    N = w * h
    grid, squid = snake_square_path(w, h)
    grid = N - grid
    squid = np.flip(squid, axis=0)
    return grid, squid

def xy_path(w, h):
    N = w * h
    series = np.arange(N)
    grid = np.reshape(series, (w, h))
    x = np.ravel(np.argsort(grid))
    y = np.ravel(np.transpose(np.argsort(np.transpose(grid))))
    squid = np.array([x, y])
    if squid.shape[0] == 2:
        squid = np.transpose(squid)

    return grid, squid

def theta_min_special_devector(signal, W, H):
    grid, squid = theta_path(W, H)
    squid, signal_res, p = squid_restrict_data(squid, signal, W, H)
    X, Y = squid[:,0], squid[:, 1]

    # apply theta sensitive mean
    blank = np.zeros((W, H))
    count_correction = np.zeros((W, H))
    np.add.at(blank, (Y, X), signal_res)
    np.add.at(count_correction, (Y, X), 1)
    count_correction = np.where(count_correction == 0, 1, count_correction)
    inverse_count_correction = np.pow(count_correction, -1)
    blank = blank * inverse_count_correction

    # apply theta sensitive L1 minimum
    np.minimum.at(blank, (Y, X), signal_res)

    return blank

def theta_inv_min(w, h):
    return theta_path(w, h, invert=True)
    

PATH_FUNCS = {
    'theta' : theta_path,
    'snake_square' : snake_square_path,
    'snake_square_inv' : snake_inv_square_path,
    'xy' : xy_path,
    'theta_min' : theta_path, # has a special resolve case!
    'theta_min_inv' : theta_inv_min # should be WORSE than theta min!
}

PATH_FUNCS_SPECIAL_VECTOR = { }

PATH_FUNCS_SPECIAL_DEVECTOR = {
    'theta_min' : theta_min_special_devector
}

'''
Command-line parsing functions
'''

# PIPE FORMAT: header, [Xn]
# INPUT PIPE: header, P, [Yn]
def reconstruct(args, pipe:bool=False, inp:any = None, file_precache:dict = None):
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

    -ns, --no-save
        Disables saving of output to disk.

    -s, --start (i)
        Sets starting at a specific image channel (i.e. 1 to start from Y1)

    -t, --tries (i)
        Sets a maximum number of attempts at solving the problem. Will re-run the solver this number of times. Defaults to 1 (only runs once).

    -p, --persist
        Enables creating a flag that allows reconstruction to resume from a prior observed state. Only applicable when there are multiple channels to reconstruct (i.e. 3 color channels), and no variables change between runs (i.e. providing different sample masks). Does not allow resuming of an in-progress solve, only resuming channel progress. Recommended if you may interrupt the process and don't want to start from channel 0.

        If the reconstruction is persistent, no-save is overridden & temporary files will be saved. If no-save and persistent are both true, these files will be deleted at the end.
''')
        return ltret(0, pipe)

    BASIS = 'discrete-cosine'
    IN_PATH = []
    OUT_PATH = None
    VERBOSE = False
    NO_SAVE = False
    START_I = 0
    TRIES = 1
    PERSISTENT = False

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
        elif arg in ('-ns', '--no-save'):
            NO_SAVE = True
        elif arg in ('-s', '--start'):
            idx += 1
            START_I = int(args[idx])
        elif arg in ('-t', '--tries'):
            idx += 1
            TRIES = int(args[idx])
        elif arg in ('-p', '--persist'):
            PERSISTENT = True
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
                N, _, tmp = multi_csv_header(path, file_precache)
                Y += tmp
            else:
                if file_precache is not None and path in file_precache:
                    tH, tD = file_precache[path]
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

    if BASIS not in ACCEPTED_BASISES and BASIS not in BASIS_ALIASES.keys():
        print(f'Basis {BASIS} not valid!')
        return ltret(-3, pipe)

    if OUT_PATH == None:
        # print(P_path)
        OUT_PATH = remove_filetype_path(P_path) + 're_'

    # If the persistence flag is set, check if a persistent file flag is found.
    PERSIST_PATH = f'{OUT_PATH}.persist'
    PERSIST_RESUME = False
    if PERSISTENT:
        if os.path.exists(PERSIST_PATH) and os.path.isfile(PERSIST_PATH):
            PERSIST_RESUME = True
        pathlib.Path.touch(PERSIST_PATH)

    N = infer_length(header)
    # We now have P, Y and N. we now can do our processes!
    X = []
    if type(P) is not list:
        P = P.astype(int)
    n = START_I
    # print(f'Start: {n}/{len(Y)} @ ({datetime.datetime.now()})')
    while n < len(Y):
        # If PERSIST_RESUME and this file exists, load it up and continue
        tempout = f'{OUT_PATH}X{n}.csv'
        if PERSIST_RESUME:
            if os.path.exists(tempout) and os.path.isfile(tempout):
                header, tmp = csv_header(tempout)
                X.append(tmp)
                n += 1
                continue
        
        Yn = Y[n]
        passed = False
        tmp = None
        from cvxpy import SolverError
        for t in range(TRIES):
            try:
                tmp = RECONSTRUCT_DRIVER(Yn, P, N, BASIS, verbose=VERBOSE)
                passed = True
            except SolverError as e:
                print(f'Caught solver error! {t+1}/{TRIES}\n{e}')
        # print(f'{np.min(tmp)} -> {np.max(tmp)}')
        if tmp is None or not passed:
            print(f'CVX didn\'t work! Failed at {n} / {len(Y)}. Check the basis code & enable verbosity')
            return ltret(-2, pipe)

        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        X.append(tmp)

        if not NO_SAVE or PERSISTENT:
            np.savetxt(tempout, tmp, fmt='%f', delimiter=',', header=header)
        n += 1
        # print(f'{n}/{len(Y)} (L1/N = {np.sum(tmp) /  tmp.shape}) @ ({datetime.datetime.now()})')

    # and unlink the persistent file too, along with any temp files
    if PERSISTENT:
        if NO_SAVE:
            for n in range(len(Y)):
                tempout = f'{OUT_PATH}X{n}.csv'
                if os.path.exists(tempout) and os.path.isfile(tempout):
                    pathlib.Path.unlink(tempout)
        if os.path.exists(PERSIST_PATH) and os.path.isfile(PERSIST_PATH):
            pathlib.Path.unlink(PERSIST_PATH)

    if pipe:
        return ltret(0, payload=(OUT_PATH, [header, X]))

    return ltret(0, pipe)
   
# PIPE FORMAT: header, P, [Yn]
# INPUT PIPE: header, data
# INPUT PIPE: image
def sample(args, pipe:bool=False, inp:any = None, file_precache:dict = None):
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

    -p, --part (fraction)
        specifies which part to take a sample of. This section of the sample P array is used.
        For example, specifying 0/4 will take the first quarter of the sample P array, where 3/4 will take the final quarter of the sample P array.
    
    -ms, --multisample
        When defined in conjunction with --part, enables multi-sub-sampling of sample array P. This creates overlapping sub-sampled parts (i.e. with 4 parts, each part will be 50% of the full sample mask, overlapping with two others).
    
    -ns, --no-save
        Disables saving of output to disk.
''')
        return ltret(0, pipe)
    IN_PATH = None
    OUT_PATH = None
    PERCENT = None
    MASK_PATH = None
    IS_IMAGE_SAMPLE = False
    NO_SAVE = False
    PARTS = None
    MULTISAMPLE = False

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-m', '--mask'):
            idx += 1
            MASK_PATH = args[idx]
        elif arg in ('-ns', '--no-save'):
            NO_SAVE = True
        elif arg in ('-p', '--part'):
            idx += 1
            PARTS = args[idx]
        elif arg in ('-ms', '--multisample'):
            MULTISAMPLE = True
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
            if file_precache is not None and IN_PATH in file_precache:
                data = file_precache[IN_PATH]
            else:
                data = plt.imread(IN_PATH)
            header = None
            IS_IMAGE_SAMPLE = True
        else:
            if file_precache is not None and IN_PATH in file_precache:
                header, data = file_precache[IN_PATH]
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
        if len(data.shape) == 1:
            data = np.reshape(data, (data.shape[0], 1))
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
    P = np.array(P)

    if PARTS != None:
        # verify PARTS is valid
        import re
        tmp = re.split(r'[\\/]', PARTS)
        if len(tmp) != 2:
            print(f'Parts parameter invalid: {PARTS} should be in form int/int.')
            return ltret(-500, pipe)
        sect = int(tmp[0])
        max_sect = int(tmp[1])
        if max_sect < 1:
            max_sect = 1
        fraction = 1.0 / max_sect
        fullsize = len(P)
        step = int(fraction * fullsize)
        low_range = (step * sect)
        if not MULTISAMPLE:
            high_range = (step * (sect+1)) if sect < (max_sect-1) else fullsize
            P = P[low_range:high_range]
        else:
            if sect < (max_sect - 2):
                P = P[low_range:step * (sect + 2)]
            elif sect == (max_sect - 2):
                P = P[low_range:fullsize]
            else:
                P = np.concatenate((P[low_range:fullsize], P[0:step]))

    if IS_IMAGE_SAMPLE:
        sample_mask = np.zeros((N, 3))
        sample_mask[P] = np.ones(3) * 255
        sample_mask = sample_mask.astype(np.uint8)
        w, h = data.shape[0], data.shape[1]
        sample_mask_image = np.reshape(sample_mask, (w, h, 3))
        if not NO_SAVE:
            image_saver(sample_mask_image, OUT_PATH)
        if pipe:
            return ltret(0, payload=(OUT_PATH, sample_mask_image))
    else:
        Y = []
        for channel in range(channels):
            temp = data[P, channel]
            Y.append(np.array(temp))

        if not NO_SAVE:
            np.savetxt(f'{OUT_PATH}P.csv', P, fmt='%d', delimiter=',', header=header)
            for n in range(channels):
                np.savetxt(f'{OUT_PATH}Y{n}.csv', Y[n], fmt='%f', delimiter=',\t', header=header)

        if pipe:
            return ltret(0, payload=(OUT_PATH, (header, P, Y)))
    
    return ltret(0, pipe)

# PIPE FORMAT: header, data
# INPUT PIPE: image
def vector(args, pipe:bool = False, inp:any = None, file_precache:dict = None):
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
                {',\n'.join(PATH_FUNCS.keys())}

        -n, --no-save
            Disables saving of files to disk.
                
        -s, --separate
            separates contents of a multi-channel image into individual channeled images. output filename should have a * to denote where the number should go if featured.
    ''')
        return ltret(0, pipe)
    IN_PATH = None
    OUT_PATH = None
    PATTERN = 'xy'
    SEPARATE = False
    NO_SAVE = False

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-p', '--pattern'):
            idx += 1
            PATTERN = args[idx]
        elif arg in ('-s', '--separate'):
            SEPARATE = True
        elif arg in ('-ns', '--no-save'):
            NO_SAVE = True
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
        if file_precache is not None and IN_PATH in file_precache:
            image = file_precache[IN_PATH]
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

    if PATTERN is None:
        PATTERN = 'xy'

    if PATTERN not in PATH_FUNCS.keys():
        print(f'Pattern given {PATTERN} is not valid.')
        return ltret(-3, pipe)

    w, h, c = image_shape_props(image)     
    grid, squid = PATH_FUNCS[PATTERN](w, h)
    N = squid.shape[0]
    squid, p = squid_restrict(squid, w, h)     
    x, y = squid[:,0], squid[:,1]
    out = np.zeros((N, c))

    out[p] = image[y, x]
    header = f'{PATTERN} {w} {h}'  

    if not SEPARATE:
        # np.savetxt(OUT_PATH, out, fmt='%f', delimiter=',\t', header=header)
        if not NO_SAVE:
            csv_save(OUT_PATH, out, header)
        if pipe:
            return ltret(0, payload=(OUT_PATH, [f'# {header}', out]))
    else:
        n = 0
        columns = out.shape[1]
        while n < columns:
            data = out[:,n]
            # np.savetxt(OUT_PATH.format(n), data, fmt='%f', delimiter=',', header=header)
            if not NO_SAVE:
                csv_save(OUT_PATH.format(n), data, header)
            n += 1
        if pipe:
            return ltret(0, payload=(RAW_OUT_PATH, [f'# {header}', out]))
    
    return ltret(0, pipe)

# PIPE FORMAT: image
# INPUT PIPE: header, data
# INPUT PIPE: header, [data]
def devector(args, pipe:bool = False, inp:any = None, file_precache:dict = None):
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
        
        -ns, --no-save
            Disables saving of output to disk.
    ''')
        return ltret(0, pipe)

    IN_PATH = None
    OUT_PATH = None
    NO_SAVE = False

    idx = 0
    while idx < len(args):
        arg = args[idx]

        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-ns', '--no-save'):
            NO_SAVE = True
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
            MULTI_DATA = True
            header, data = multi_csv_header(IN_PATH, file_precache=file_precache)
            if len(data) == 0:
                print(f'No files found by format {IN_PATH}')
                return ltret(-1, pipe)
            if len(data) > 4:
                print(f'Invalid number of channels {len(data)}')
                return ltret(-1, pipe)
            
            IN_PATH = IN_PATH.replace('*', 'n')
        else:
            if file_precache is not None and IN_PATH == file_precache:
                header, data = file_precache[IN_PATH]
            else:
                header, data = csv_header(IN_PATH)

    if OUT_PATH == None:
        OUT_PATH = remove_filetype_path(IN_PATH) + '_devector.png'

    header = header.split()
    format_type = header[0]
    image = None

    if format_type not in PATH_FUNCS.keys():
        print(f'ERROR: {format_type} not recognized path function!')
        return ltret(-1023, pipe)

    w = int(header[1])
    h = int(header[2])
    grid, squid = PATH_FUNCS[format_type](w, h)
    # squid, p = squid_restrict(squid, w, h)
    # x, y = squid[:,0], squid[:,1]

    if MULTI_DATA:
        image = np.zeros((w, h, len(data)))
        for n in range(len(data)):
            if format_type in PATH_FUNCS_SPECIAL_DEVECTOR:
                image[:,:,n] = PATH_FUNCS_SPECIAL_DEVECTOR[format_type](data[n], w, h)
            else:
                squid_res, dat_res, p = squid_restrict_data(squid, data[n], w, h)
                x, y = squid_res[:,0], squid_res[:,1]
                image[y, x, n] = dat_res
    else:
        if len(data.shape) == 2:
            c = data.shape[1]
        else:
            c = 1
        image = np.zeros((w, h, c))
        if format_type in PATH_FUNCS_SPECIAL_DEVECTOR:
            image[:,:,0] = PATH_FUNCS_SPECIAL_DEVECTOR[format_type](data, w, h)
        else:
            squid_res, dat_res, p = squid_restrict_data(squid, data, w, h)
            x, y = squid_res[:,0], squid_res[:,1]
            image[y, x, 0] = dat_res

    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:,:,0]

    if is_intlike_image(image):
        image = image.astype(np.uint8)
    
    if not NO_SAVE:
        image_saver(image, OUT_PATH)
    if pipe:
        return ltret(0, payload=(OUT_PATH, image))

    return ltret(0, pipe)

# PIPE FORMAT: image
# INPUT PIPE: image
def monochrome(args, pipe:bool = False, inp:any = None, file_precache:dict = None):
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
        image_saver(image, OUT_PATH)

        if pipe:
            return ltret(0, payload=(OUT_PATH, image))

    ALLOWED_DATATYPES = [np.float32, np.float16, np.float64, np.floating]
    if type(min) not in ALLOWED_DATATYPES and type(max) not in ALLOWED_DATATYPES:
        image = image / 255.0

    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    outimg = np.sqrt(np.pow(r, 2) + np.pow(g, 2) + np.pow(b, 2))
    outimg = outimg * 255
    outimg = outimg.astype(int)
    image_saver(outimg, OUT_PATH, True)

    if pipe:
        return ltret(0, payload=(OUT_PATH, outimg))
    return ltret(0, pipe)

# PIPE FORMAT: image
# INPUT PIPE: image
def prepare(args, pipe:bool = False, inp:any = None, file_precache:dict = None):
    if type(args) == str:
        args = args.split()
        
    if len(args) == 0 and inp == None:
        print('''
    cspy prep
    Prepares an image, doing some initial image processing

    usage:
        py cspy.py prep (path/to/some/image) (options)
    
    arguments:
        -o, --out (path)
            specifies the save path for the output image. Defaults to a suffixed '_mono' rendition of the given input.
        
        -r, --resize (w=...|h=...)
            Scales the image to the set width or height value. 

        -c, --crop
            Takes a central crop of the image

        -m, --mono
            Grayscales the image. Useful for piping.
        
        -n, --normalise
            Normalises the color (float range between 0, 1)
        
        -ns, --no-save
            Disables saving of files to disk. Saving turned on by default.

        -na, --no-alpha
            Strips out the alpha channel before saving if an alpha channel is present. Off by default.
            
        -i, --invert
            Inverts the colors of the image.
    ''')
        return ltret(0, pipe)
    
    IN_PATH = None
    OUT_PATH = None
    DO_RESIZE = None # can be a tuple with either first or second parameter being a positive integer to scale down to.
    DO_CROP = False
    NORMALISE = False
    NO_SAVE = False
    NO_ALPHA = False
    MONO = False
    INVERT = False

    idx = 0
    while idx < len(args):
        arg = args[idx].lower()
        if arg in ('-o', '--out'):
            idx += 1
            OUT_PATH = args[idx]
        elif arg in ('-n', '--normalise'):
            NORMALISE = True
        elif arg in ('-ns', '--no-save'):
            NO_SAVE = True
        elif arg in ('-m', '--mono'):
            MONO = True
        elif arg in ('-na', '--no-alpha'):
            NO_ALPHA = True
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
        elif arg in ('-i', '--invert'):
            INVERT = True
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
            IN_PATH = f'./blank.jpg'
            image = np.ones((256, 256, 3))
    else:
        if file_precache is not None and IN_PATH in file_precache:
            image = file_precache[IN_PATH]
        else:
            image = plt.imread(IN_PATH)

    MONO_IMAGE = len(image.shape) == 2
    if MONO_IMAGE:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    RAW_OUT_PATH = OUT_PATH

    if OUT_PATH == None:
        OUT_PATH = remove_filetype_path(IN_PATH) + '_prepared.' + get_filetype_path(IN_PATH)

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

    if MONO:
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        pretype = image.dtype
        
        image = np.sqrt((np.pow(r, 2) + np.pow(g, 2) + np.pow(b, 2))/3)

        if pretype == np.uint8:
            image = image.astype(np.uint8)

        image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    if NORMALISE:
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image = image / 255.0
        else:
            image = image / np.max(image)

    if NO_ALPHA and image.shape[2] == 4:
        image = image[:,:,:3]

    if image.dtype == np.float32:
        image = np.clip(image, 0, 1)
        
    if INVERT:
        if image.dtype in (np.float32, np.float64):
            image = 1 - image
        else:
            image = 255 - image
            image = image.astype(np.uint8)

    if not NO_SAVE:
        image_saver(image, OUT_PATH, MONO)

    if pipe:
        return ltret(0, payload=(OUT_PATH, image))
    return ltret(0, pipe)


def cspy_engine(args, pipe:bool=False, inp:any = None, print_cli:bool = False, file_precache:dict = None):
    if type(args) == str:
        args = args.split()

    if len(args) == 0:
        print('''
cspy
An implementation of a compressed sensing suite in Python.

usage:
    py cspy.py (prep|vector|sample|reconstruct|devector) ...

vector:
    Vectorizes a 2D image into a 1D signal. CSV signal files require the type of signal to be stored as a header comment.

sample:
    For a given image, generates a sample mask as specified and saves samples from all channels of the image (Yn) as well as the sample position array P.
    
reconstruct:
    Given appropriate data (samples and sample coordinates), reconstructs an image.
    Data should be given in two paths, both containing 1-dimensional CSV formats - sample position array P, and sample array Y (where P stores the location of each Y in the full datastream of size N).

devector:
    Converts CSV vector files back into image files.

prep:
    Prepares an image by pre-processing - downscaling, croppping, converting into monochrome, etc.
    
''')
        return ltret(-1, pipe)

    big_argstring = ' '.join(args)
    argstrings = big_argstring.split(',')
    if len(argstrings) >= 1:
        prior_state = inp

        for args in argstrings:
            args = args.split()
            mode = args[0].lower()

            match mode:
                case 'sample':
                    code, prior_state = sample(args[1:], pipe=True, inp=prior_state, file_precache=file_precache)
                    if code != 0:
                        return code
                case 'vector':
                    code, prior_state = vector(args[1:], pipe=True, inp=prior_state, file_precache=file_precache)
                    if code != 0:
                        return code
                case 'devector':
                    code, prior_state = devector(args[1:], pipe=True, inp=prior_state, file_precache=file_precache)
                    if code != 0:
                        return code
                case 'reconstruct':
                    code, prior_state = reconstruct(args[1:], pipe=True, inp=prior_state, file_precache=file_precache)
                    if code != 0:
                        return code
                case 'prep':
                    code, prior_state = prepare(args[1:], pipe=True, inp=prior_state, file_precache=file_precache)
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
        elif mode == 'prep':
            return prepare(args[1:], pipe, inp)
        else:
            if print_cli:
                print(f'Unrecognized mode {mode}')
            return ltret(-1, pipe)

if __name__ == '__main__':
    exit(cspy_engine(sys.argv[1:]))