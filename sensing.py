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

def test_single(path:str, size:int=75, sample_percent:float = 0.1):
    image = plt.imread(path)
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    print(f'image is {image.shape} along first 2 axis')

    # crop image down to central square
    image = square_crop(image)
    raw_image = image
    focus = size / image.shape[0] 

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

    mask_img = np.zeros(num_points)
    mask_img[samples] = 1
    mask_img = np.resize(mask_img, (axis_size, axis_size))
    mask_img = swizzle_image(condense_frames([mask_img, mask_img, mask_img]))
    mask_img = mask_img * image
    
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask_img)
    plt.show()

    plt.figure(figsize=(10, 10))

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
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        plt.subplot(nrows, ncols, (idx*ncols)+2)
        plt.imshow(frames[idx], cmap=democolors)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        plt.subplot(nrows, ncols, (idx*ncols)+3)
        plt.imshow(np.abs(reconstructed[idx] - frames[idx]), cmap=democolors)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        plt.subplot(nrows, ncols, (idx*ncols)+4)
        plt.hist(data-recon, bins=25)
        # ax = plt.gca()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        idx += 1

    recon_image = swizzle_image(condense_frames(reconstructed))
    recon_image = (recon_image - np.min(recon_image)) / (np.max(recon_image) - np.min(recon_image))

    plt.subplot(nrows, ncols, (nrows-1)*ncols + 1)
    plt.imshow(recon_image, interpolation='bicubic')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    plt.subplot(nrows, ncols, (nrows-1)*ncols + 2)
    plt.imshow(image, interpolation='bicubic')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.subplot(nrows, ncols, (nrows-1)*ncols + 3)
    plt.imshow(raw_image)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.subplot(nrows, ncols, (nrows-1)*ncols + 4)
    plt.imshow(mask_img)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    plt.tight_layout()
    plt.savefig(f'{path}.cmp.png')
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(recon_image)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}.out.png')
    
    plt.imsave(f'{path}.raw.png', recon_image)
    
    print(f'Saved outputs')

def test_grayscale(path:str, size:int=75, sample_percent:float=0.25):
    image = plt.imread(path)
    image = square_crop(image)
    # image = unswizzle_image(image)
    image = make_gray(image)

    axis_size = image.shape[0]
    scale_factor = size / axis_size
    image = np.clip(zoom(image, scale_factor), 0, 255).astype('float') / 256
    axis_size = image.shape[0]
    
    # make raw data and pick samples
    raw_data = np.ravel(image)
    num_points = len(raw_data)
    sample_count = int(sample_percent * num_points)
    samples = take_samples(num_points, sample_count)
    
    # generate visual mask
    mask_image = np.zeros(num_points)
    mask_image[samples] = 1
    mask_image = np.resize(mask_image, (axis_size, axis_size)) * image
    
    print(f'max,min=({np.max(image), np.min(image)})\ndimensions {image.shape}\n{num_points} points ({sample_percent*num_points} samples)')
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask_image, cmap='gray')
    plt.show()
    
    reconstruction = reconstruct(raw_data[samples], num_points, samples, mode='cvx')
    
    recon_image = np.resize(reconstruction, (axis_size, axis_size))
    
    residual = np.abs(recon_image - image)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(recon_image, cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplot(1, 3, 3)
    plt.imshow(residual, cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig(f'{path}.grayscale.cmp.jpg')
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(recon_image)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}.grayscale.out.png')
    
def test_anychannels(path:str, size:int=50, sample_percent:float=0.25, bincount=25, cmap='viridis', force_data=None, normalise_recon:bool = False):
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
    
    plt.figure(figsize=(8, 4), facecolor=(0, 0, 0, 0))
    plt.subplot(1,2,1)
    plt.imshow(cropped_image)
    plt.title('Cropped & scaled image')
    plt.subplot(1,2,2)
    plt.imshow(sample_mask, cmap=cmap)
    plt.title(f'Sample selection from cropped image')
    plt.show()
    
    plt.figure(figsize=(12, 12))
    nrows = channels + 1
    ncols = 4 # image, reconstruction, residual, histogram residual
    is_in_first = True
    reconstructions = []
    
    for channel_idx in range(channels):
        channel = cropped_image[:,:,channel_idx]
        channel_ravel = np.ravel(channel)
        
        recon = reconstruct(channel_ravel[samples], full_size, samples, mode='cvx')
        if normalise_recon:
            recon = recon - np.min(recon) / (np.max(recon) - np.min(recon))
        
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
        

def main():
    test_anychannels(f'./testdata/space.jpg', size=50, sample_percent=0.50, normalise_recon=True)
    # junk = np.zeros((100, 100, 3))
    # for x in range(0, 100):
    #     for y in range(0, 100):
    #         z = 142 - np.sqrt(x**2 + y**2)
    #         R = x/100.0
    #         G = y/100.0
    #         B = z/142.0
    #         junk[x,y] = np.array([R, G, B])
    # test_anychannels(f'./testdata/hue', size=50, sample_percent=0.2, force_data=junk, normalise_recon=True)
    pass

if __name__ == '__main__':
    main()