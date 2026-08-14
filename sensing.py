import os, sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
import cvxpy as cp

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

def reconstruct(y, n, p, sparsity=5, epsilon=1e-10, max_iter=1000, mode='cosamp'):
    '''
    for a sampling y, with n maximum spaces in the dataset and sample vector p, computes and returns the reconstructed signal x'''
    psi = dct(np.eye(n, n))
    theta = psi[p, :]
    if mode=='cosamp':
        s = cosamp(theta, y, sparsity, epsilon, max_iter)
    elif mode=='cvx':
        s_l1 = cp.Variable(n)
        obj = cp.Minimize(cp.norm(s_l1, 1))
        cons = [theta @ s_l1 == np.conjugate(y)]
        problem = cp.Problem(obj, cons)
        problem.solve(verbose=True)
        s = s_l1.value


    return idct(s)

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
    # inverts a 3, X, Y image into X, Y, 3
    a, b, c = image
    
    size = a.shape[0] # should be consistent
    a, b, c = np.ravel(a), np.ravel(b), np.ravel(c)
    
    outp = [ [i, j, k] for i, j, k in zip(a, b, c)]
    outp = np.array(outp)
    outp = np.reshape(outp, (size, size, 3))
    return outp

def unswizzle_image(image):
    # converts an XY3 image into 3XY
    image_rgbs = [image[:,:,i] for i in range(3)]
    raw = np.ndarray((3, image.shape[0], image.shape[1]))
    for i in range(3):
        raw[i] = image_rgbs[i]
    return raw

def decondense_frames(image):
    # decondenses an swizzled image into a list of 3 frames
    return [image[:,:,i] for i in range(3)]

def condense_frames(frames):
    # condenses multiple frames into unswizzled format
    raw = np.ndarray((3, frames[0].shape[0], frames[0].shape[1]))
    for i in range(3):
        raw[i] = frames[i]
    return raw

def is_swizzled(image):
    '''
    returns true if image is swizzled (in XY3 format), false if unswizzled (3XY)'''
    return image.shape[2] == 3

def square_crop(image):
    swizzled=False
    if not is_swizzled(image):
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

'''
Main sequence code
'''

def test_single(path:str, focus:float=0.125, sample_percent:float = 0.1):
    image = plt.imread(path)
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    print(f'image is {image.shape} along first 2 axis')

    # crop image down to central square
    image = square_crop(image)
    raw_image = image

    # unswizzle it so we can resize each frame
    image_rgb = unswizzle_image(image)
    frames = [np.clip(zoom(frame, focus), 0, 255).astype('int') for frame in image_rgb]
    frames = [(frame / 255).astype('float') for frame in frames]

    # comparison image in full color - frames keeps each individual image
    image = condense_frames(frames)
    image = swizzle_image(image)

    num_points = image.shape[0] * image.shape[1]
    datas = [np.ravel(frame) for frame in frames]
    axis_size = image.shape[0]

    print(f'max,min=({np.max(image), np.min(image)})\ndimensions {image.shape}\n{num_points} points ({sample_percent*num_points} samples)')

    SAMPLE_COUNT = int(sample_percent * num_points)
    idx = 0
    reconstructed = [[], [], []]
    
    samples = take_samples(num_points, SAMPLE_COUNT)

    plt.figure(figsize=(8, 8))

    nrows, ncols = 4, 4

    democolors = 'Grays'
    for data in datas:
        length = len(data)
        # print(samples)
        recon = reconstruct(data[samples], length, samples, mode='cvx')
        print(f'{idx+1}/3')

        reconstructed[idx] = np.resize(recon, (axis_size, axis_size))
        plt.subplot(nrows, ncols, (idx*ncols)+1) 
        plt.imshow(reconstructed[idx], cmap=democolors)
        plt.subplot(nrows, ncols, (idx*ncols)+2)
        plt.imshow(frames[idx], cmap=democolors)
        plt.subplot(nrows, ncols, (idx*ncols)+3)
        plt.imshow(np.abs(reconstructed[idx] - frames[idx]), cmap=democolors)
        plt.subplot(nrows, ncols, (idx*ncols)+4)
        plt.plot(data-recon)
        idx += 1

    recon_image = swizzle_image(condense_frames(reconstructed))
    recon_image = (recon_image - np.min(recon_image)) / (np.max(recon_image) - np.min(recon_image))

    plt.subplot(nrows, ncols, (nrows-1)*ncols + 1)
    plt.imshow(recon_image, interpolation='bicubic')
    plt.subplot(nrows, ncols, (nrows-1)*ncols + 2)
    plt.imshow(image, interpolation='bicubic')
    plt.subplot(nrows, ncols, (nrows-1)*ncols + 3)
    plt.imshow(raw_image)

    plt.tight_layout()
    plt.show()



def main():
    test_single(f'./testdata/space.jpg', focus=0.04, sample_percent = 0.6)

if __name__ == '__main__':
    main()