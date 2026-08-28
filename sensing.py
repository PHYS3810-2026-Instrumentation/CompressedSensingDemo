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

'''
Misc. data structures
'''
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

'''
Below are functions pertinent to deconstructing & reconstructing images (a CoSaMP implementation included)

'''

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
    returns a list of sampled indicies for X, such that x[indicies] returns only those samples'''
    selections = np.round(np.random.rand(samples, 1) * length-1)
    selections = [int(e[0]) for e in selections]
    return selections

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

    # TODO 
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

'''
Below are functions related to image processing (swizzling in particular)
'''

def plot_nice_image(img, pow=None, darkBack = True, invert=False, cmap:tuple = None):
    cs = ('Reds', 'Greens', 'Blues')
    if cmap != None:
        cs = cmap
    if darkBack:
        backing = np.ones(shape=(img.shape[0], img.shape[0]))
        plt.imshow(backing, cmap='gray')
    for n in range(3):
        imgsel = img[:,:,n]
        if darkBack:
            achannel = img[:,:,n]
            achannel = np.where(achannel > 0, achannel, 0)
            if pow != None:
                if type(pow) in (int, float):
                    achannel = np.pow(achannel, pow)
                else:
                    achannel = np.pow(achannel, pow[n])
            if invert:
                imgsel = 1-imgsel
            plt.imshow(imgsel, cmap=cs[n], alpha=achannel)
        else:
            if invert:
                imgsel = 1-imgsel
            plt.imshow(imgsel, cmap=cs[n], alpha=0.5)

def swizzle_image(image):
    # inverts a n, X, Y image into X, Y, n
    n = image.shape[0]
    size = image.shape[1]
    channels = [image[c,:,:] for c in range(n)]
    # a, b, c = image
    
    # size = a.shape[0] # should be consistent
    # a, b, c = np.ravel(a), np.ravel(b), np.ravel(c)
    
    channels_ravel = [np.ravel(e) for e in channels]
    fullsize = len(channels_ravel[0])
    
    # outp = [ [i, j, k] for i, j, k in zip(a, b, c)]
    outp = [ [channels_ravel[elem][outer_idx] for elem in range(len(channels_ravel))] for outer_idx in range(fullsize)]
    outp = np.array(outp)
    outp = np.reshape(outp, (size, size, n))
    return outp

def unswizzle_image(image, cc:int=3):
    # converts an XY3 image into 3XY
    image_rgbs = [image[:,:,i] for i in range(3)]
    raw = np.ndarray((3, image.shape[0], image.shape[1]))
    for i in range(3):
        raw[i] = image_rgbs[i]
    return raw

def decondense_frames(image, cc:int=3):
    # decondenses an swizzled image into a list of 3 frames
    return [image[:,:,i] for i in range(3)]

def condense_frames(frames, cc:int=3):
    # condenses multiple frames into unswizzled format
    raw = np.ndarray((cc, frames[0].shape[0], frames[0].shape[1]))
    for i in range(cc):
        raw[i] = frames[i]
    return raw

def is_swizzled(image):
    '''
    returns true if image is swizzled (in XY3 format), false if unswizzled (3XY)'''
    return image.shape[2] == 3

def square_crop(image, check_swizzle=True):
    swizzled=False
    if check_swizzle and not is_swizzled(image):
        image = swizzle_image(image)
        swizzled = True

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

    if swizzled:
        image = unswizzle_image(image)
    return image

def make_gray(rgb):
    # turns (x,y,3) into (x,y,1)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray

def to_jpg(png):
    image = plt.imread(png)
    plt.imsave(f'{png}.jpg', image)

def no_axis():
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

'''
Main sequence code
'''

# @jit(nopython=True)
def test_anychannels(path:str, size:int=50, sample_percent:float=0.25, bincount=25, cmap='viridis', force_data=None, normalise_recon:bool = False, test_vis:bool = False, basis:Basis = Basis.DISCRETE_COSINE, take_real:bool=True, verbose:bool = False):
    if force_data is not None:
        image = force_data
    else:
        image = plt.imread(path)
    image = square_crop(image, check_swizzle=False)
    channels = image.shape[2]
    min, max = np.min(image), np.max(image)
    print(f'Image is of shape {image.shape} ({channels} channels) with minmax of ({min}, {max}) ({type(min)}, {type(max)})')
    ALLOWED_DATATYPES = [np.float32, np.float16, np.float64, np.floating]
    if type(min) not in ALLOWED_DATATYPES and type(max) not in ALLOWED_DATATYPES:
        print(f'Normalising to float in 0.0-1.0')
        image = image / 255.0
    
    presize = image.shape[0]
    factor = size/presize
    
    indiv = [zoom(image[:,:,e], factor) for e in range(channels)]
    mean_size = int(np.mean([e.shape for e in indiv]))
    cropped_image = np.reshape(np.array(indiv), (channels, mean_size, mean_size))
    cropped_image = np.clip(cropped_image, 0, 1)
    cropped_image = condense_frames(cropped_image, cc=channels)
    cropped_image = swizzle_image(cropped_image)
    
    print(f'Resized now {cropped_image.shape}')
    
    axis_size = cropped_image.shape[0]
    full_size = axis_size ** 2
    sample_count = int(sample_percent * full_size)
    
    samples = take_samples(full_size, sample_count)
    sample_mask = np.zeros((full_size, channels))
    sample_mask[samples] = np.ones(channels)
    sample_mask = np.reshape(sample_mask, (axis_size, axis_size, channels))
    sample_mask = sample_mask * cropped_image
    
    plt.figure(figsize=(12, 12))
    nrows = channels + 1
    ncols = 4 # image, reconstruction, residual, histogram residual
    is_in_first = True
    reconstructions = []
    
    for channel_idx in range(channels):
        channel = cropped_image[:,:,channel_idx]
        channel_ravel = np.ravel(channel)
        
        recon = reconstruct(channel_ravel[samples], full_size, samples, mode='cvx', basis=basis, verbose=verbose)
        # check that its not complex, if complex, take depending on take_real
        COMPLEXES = [np.complex64, np.complex128]
        if take_real:
            recon = np.real(recon)
        else:
            recon = np.imag(recon)
            
        print(f'Got recon of len {len(recon)} (min, max = {np.min(recon)}, {np.max(recon)})')
        # recon = [(e.real if take_real else e.imag) if (type(e) in COMPLEXES) else e for e in recon]
        
        if normalise_recon:
            recon = (recon - np.min(recon)) / (np.max(recon) - np.min(recon))
            print(f'(normalised to min, max = {np.min(recon)}, {np.max(recon)})')
        
        recon_channel = np.reshape(recon, (axis_size, axis_size))
        residual = np.abs(recon - channel_ravel)
        residual_channel = np.reshape(residual, (axis_size, axis_size))
        
        plt.subplot(nrows, ncols, (channel_idx * ncols) + 1)
        if is_in_first:
            plt.title('Raw image channel')
        plt.imshow(channel, cmap=cmap)
        no_axis()
        
        plt.subplot(nrows, ncols, (channel_idx * ncols) + 2)
        if is_in_first:
            plt.title('Reconstructed channel')
        plt.imshow(recon_channel, cmap=cmap)
        no_axis()
        
        plt.subplot(nrows, ncols, (channel_idx * ncols) + 3)
        if is_in_first:
            plt.title('Residual in channel')
        plt.imshow(residual_channel, cmap=cmap)
        no_axis()
        
        plt.subplot(nrows, ncols, (channel_idx * ncols) + 4)
        if is_in_first:
            plt.title('Residual dispersion')
        plt.hist(residual, bins=bincount)
        
        is_in_first = False
        reconstructions.append(recon)
        
    root_pos = channels * 4
    reconstructed_image = swizzle_image(condense_frames([np.reshape(e, (axis_size, axis_size)) for e in reconstructions], channels))
    residual_image = np.abs(cropped_image - reconstructed_image)
    
    plt.subplot(nrows, ncols, root_pos + 1)
    plt.title('Full image')
    plt.imshow(cropped_image)
    no_axis()
    
    plt.subplot(nrows, ncols, root_pos + 2)
    plt.title('Reconstructed image')
    plt.imshow(reconstructed_image)
    no_axis()
    
    plt.subplot(nrows, ncols, root_pos + 3)
    plt.title('Residual image')
    plt.imshow(residual_image)
    no_axis()
    
    plt.subplot(nrows, ncols, root_pos + 4)
    plt.title('Selection mask')
    plt.imshow(sample_mask)
    no_axis()
    plt.tight_layout()
    plt.savefig(f'{path}.cmp.png')
    
    plt.figure(figsize=(10, 10), facecolor=(0, 0, 0, 0))
    plt.imshow(reconstructed_image)
    no_axis()
    plt.tight_layout()
    plt.savefig(f'{path}.out_tr.png', transparent=True)
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(reconstructed_image)
    no_axis()
    plt.tight_layout()
    plt.savefig(f'{path}.out_op.png', transparent=False)       

def main(args):
    test_anychannels(f'./testdata/ocean.jpg', size=25, sample_percent=0.5, normalise_recon=True, test_vis=False, basis=Basis.FOURIER, verbose=True)

if __name__ == '__main__':
    main(sys.argv)