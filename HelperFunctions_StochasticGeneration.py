import numpy as np
from itertools import product
from scipy.stats import norm
import random
from queue import Queue
"""
As a word of warning: this is research code. It *should* work. But who knows. 

In general, the 2D code has been debugged significantly more rigorously than the 3D code. 

Also, I am going to leave some random code in here. For example, there is code to do the following:

(1) Compute statistics (auto and cross)
(2) Generate random shape based microstructures
(3) Compute periodic Chord Length Distributions for 2D microstructures in the X and Y direction
(4) Compute expected chordlength averaged over all directions using Berryman's method
(5) Floodfilling code for approximately extracting the cluster function from autocorrelations


Contact: arobertson38@gatech.edu
"""
def zoomNphase(arr, factor, dim=2):
    '''
    This is a method that increases the resolution of a N-phase image by subdividing each voxel into factor**dim number
    of voxels.

    :param arr: The array that we are zooming in on. The last dimension must be number of phases
    :param factor: The number of voxels we are subdividing by
    :param dim: the number of dimensions of the image
    :return:
    '''
    assert len(arr.shape) == dim+1
    assert type(factor) == int
    assert factor >= 1

    shape = [int(factor*i) for i in arr.shape[:-1]]
    shape.append(arr.shape[-1])
    new = np.zeros(shape)

    for i in range(arr.shape[-1]):
        new[..., i] = zoom(arr[..., i], factor, dim)

    return new

def zoom(arr, factor, dim=2):
    '''
    This is a method that increases the resolution of a 1-phase image by subdividing each voxel into factor**dim number
    of voxels.

    This will be written to work on only an single phase image

    :param arr: The array that we are zooming in on
    :param factor: The number of voxels we are subdividing by
    :param dim: the number of dimensions of the image
    :return:
    '''
    assert len(arr.shape) == dim
    assert factor >= 1
    assert type(factor) == int

    if dim == 2:
        new = np.zeros([arr.shape[0] * factor, arr.shape[1] * factor])
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                if arr[i, j] != 0:
                    new[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = arr[i, j]
    elif dim == 3:
        new = np.zeros([arr.shape[0] * factor, arr.shape[1] * factor], arr.shape[2] * factor)
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                for k in range(0, arr.shape[2]):
                    if arr[i, j, k] != 0:
                        new[i*factor:(i+1)*factor, j*factor:(j+1)*factor, k*factor:(k+1)*factor] = arr[i, j, k]
    else:
        raise AttributeError

    return new

def p2_crosscorrelation(arr1, arr2):
    """
    defines the crosscorrelation between arr1 and arr2:
    :param arr1:
    :param arr2:
    :return:
    """
    ax = list(range(0, len(arr1.shape)))
    arr1_FFT = np.fft.rfftn(arr1, axes=ax)
    arr2_FFT = np.fft.rfftn(arr2, axes=ax)
    return np.fft.irfftn(arr1_FFT.conjugate() * arr2_FFT, s=arr1.shape, axes=ax).real / np.product(
        arr1.shape)

def twopointstats(struct):
    """
    THis method computes and returns the full two point statistics
    :param str:
    :return:
    """
    stats = np.zeros_like(struct, dtype=np.float64)
    for i in range(0, struct.shape[-1]):
        stats[..., i] = p2_crosscorrelation(struct[..., 0], struct[..., i])
    return stats

def autocorrelations(str):
    """
    THis method computes and returns the complete set of autocorrelations
    :param str:
    :return:
    """
    assert str.shape[-1] < 15
    stats = np.zeros_like(str)
    for i in range(0, str.shape[-1]):
        stats[..., i] = p2_crosscorrelation(str[..., i], str[..., i])
    return stats

# -------------------------------------------------------------------------------
# This section includes methods for computing derivatives
# -------------------------------------------------------------------------------

def tf_x_derivative(short=1.0, long=np.sqrt(2)):
    """
    I think, because matplotlib displays images weirdly, the x and y axes are inverted. 
    """
    return (-1/4) * np.array([[0, 0, -1/short, 0, 0],
                              [0, 0, 4/short, 0, 0],
                              [0, 0, -6/short, 0, 0],
                              [0, 0, 4/short, 0, 0],
                              [0, 0, -1/short, 0, 0]])

def tf_y_derivative(short=1.0, long=np.sqrt(2)):
    """
    I think, because matplotlib displays images weirdly, the x and y axes are inverted. 
    """
    return (-1/4) * np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [-1/short, 4/short, -6/short, 4/short, -1/short],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])

def two_forward_derivative(short=1.0, long=np.sqrt(2)):
    return (-1/16) * np.array([[-1/long, 0, -1/short, 0, -1/long],
                              [0, 4/long, 4/short, 4/long, 0],
                              [-1/short, 4/short, -12/short - 12/long, 4/short, -1/short],
                              [0, 4/long, 4/short, 4/long, 0],
                              [-1/long, 0, -1/short, 0, -1/long]])

def approx_matrix_derivative(arr, matrix_deriv=two_forward_derivative, short_length=1.0, long_length=np.sqrt(2)):
    """
    assumes that the field enters uncentered
    """
    sh = arr.shape
    cent = np.fft.fftshift(arr)[int(sh[0] / 2 - 2):int(sh[0] / 2 + 3), int(sh[1] / 2 - 2):int(sh[1] / 2 + 3)]
    return (cent * matrix_deriv(short_length, long_length)).sum()

# -------------------------------------------------------------------------------
# 2D Chord Length Distribution Code
# -------------------------------------------------------------------------------

def _cld2D(image, bins):
    """
    This code is the backend for the CLD code provided below

    it passes only the x direction
    """
    chords = []
    x_max = image.shape[0]
    for y in range(image.shape[1]):
        x = 0
        flag = True
        while x < x_max:
            if image[x, y] > 0:
                if x==0:
                    if image[-1, y] > 0:
                        flag = False
                x_end = x+1
                while image[x_end%x_max, y] > 0:
                    if x_end%x_max == x:
                        # then we have found a while line of one phase
                        flag = True
                        break
                    x_end += 1
                if flag:
                    # add the chord length to the list.
                    chords.append(x_end - x)
                x = x_end
                flag = True
            else:
                x += 1
    if type(bins) != type(True):
        chords = list(np.histogram(chords, bins=bins)[0])
    return chords


def cld2D(image, bins, axis=0):
    """
    this is code that computes the chord length distribution for 2D PERIODIC microstructures along
    either the X or Y directions. 

    This uses numpy convention. Therefore, x is the visual y

    axis == 0 => the x direction
    axis == 1 => the y direction
    axis == -1 => both returns [x_chords, y_chords]
    """
    assert len(image.shape) == 2

    if axis == 0:
        return _cld2D(image, bins)
    elif axis == 1:
        return _cld2D(image.transpose(), bins)
    elif axis == -1:
        return [_cld2D(image, bins), _cld2D(image.transpose(), bins)]
    else:
        raise AttributeError("Invalid axis. 0: numpy x, 1: numpy y, -1: both")

# -------------------------------------------------------------------------------
# Generating Shape Based Microstructures
# -------------------------------------------------------------------------------

def square(size):
    return np.ones([int(2*size+1), int(2*size+1)])

def diagonal(size):
    assert size > 0
    a = np.ones(int(size*2 + 1))
    b = np.ones(int(size*2))
    return np.diag(a,0) + np.diag(b, 1) + np.diag(b, -1)

def X(size):
    a = diagonal(size) 
    a = a + a[:, ::-1]
    a[a>0.0] = 1.0
    return a

def diamond(size):
    shape = np.zeros([int(2*size+1), int(2*size+1)])
    for i in range(2*size+1):
        for j in range(2*size+1):
            if np.abs(i-size) + np.abs(j-size) <= size:
                shape[i, j] = 1.0
    return shape

def circle(size):
    shape = np.zeros([int(2*size+1), int(2*size+1)])
    for i in range(2*size+1):
        for j in range(2*size+1):
            if np.square(i-size) + np.square(j-size) <= np.square(size):
                shape[i, j] = 1.0
    return shape

def shape_structure_generation(dim=100, choices=[1,2,3,4,5,6], number_of_cubes_placed=30, max_iter=40, shape=square):
    center_x = []
    center_y = []
    collected_sizes = []
    iter = 0
    placed = 0
    struct = np.zeros([dim, dim])
    loc_choice = list(range(0, dim))
    while placed < number_of_cubes_placed:
        while iter < max_iter:
            size = random.choice(choices)
            x_select = random.choice(loc_choice[size:(-1*size)])
            y_select = random.choice(loc_choice[size:(-1*size)])
            if np.all(np.logical_or(np.abs(np.array(center_x) - x_select) > (size + 1 + np.array(collected_sizes)), \
                    np.abs(np.array(center_y) - y_select) > (size + 1 + np.array(collected_sizes)))):
                struct[(x_select-(size)):(x_select+size+1), (y_select-(size)):(y_select+size+1)] = shape(size)
                center_x.append(x_select)
                center_y.append(y_select)
                collected_sizes.append(size)
                iter = max_iter
                placed += 1
            else:
                iter += 1
        iter = 0
    return struct, collected_sizes

# -------------------------------------------------------------------------------
# Methods for extracting the center of autocorrelations
# -------------------------------------------------------------------------------

def _disect_floodfill(arr, cutoff=0.1, radius_cutoff_fraction=0.33, return_mask=False):
    """
    The user *should* input the covariance function, not the autocorrelation

    This will remove the center of the covariance.

    We assume that the arr is input centered (no np.fft.fftshift is necessary)
    
    Array is output centered, along with the mask and the longest direction

    We can also use the space inversion symmetry of the covariance to half the 
    number of computations. 

    :param arr:
    :param cutoff:
    :return:
    """
    
    arr_max = arr.max() * cutoff
    size = min(arr.shape)
    radius_cutoff = radius_cutoff_fraction * size
    xmax = arr.shape[0]
    ymax = arr.shape[1]

    centx = int(xmax / 2)
    centy = int(ymax / 2)

    flags = np.zeros_like(arr)
    flags[centx, centy] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [(centx - 1, centy), (centx, centy + 1), (centx + 1, centy)]]

    maxup = 0
    maxdown = 0
    maxleft = 0
    maxupright = 0
    maxdownright = 0


    while not voxel_queue.empty():
        x, y = voxel_queue.get()
        direc = np.linalg.norm(np.array([x, y]) - np.array([centx, centy]))

        if arr[x, y] > arr_max and direc < radius_cutoff and flags[x, y] != 1 and x>-1 and y >= centy and x < xmax and y < ymax:
            if x == centx:
                maxleft = max(maxleft, direc)
            elif y == centy and x < centx:
                maxdown = max(maxdown, direc)
            elif y == centy and x > centx:
                maxup = max(maxup, direc)
            elif y == x:
                maxupright = max(maxupright, direc)
            elif abs(y - centy) == abs(x - centx):
                maxdownright = max(maxdownright, direc)
                
            flags[x, y] = 1.0
            voxel_queue.put((x+1, y))
            voxel_queue.put((x-1, y))
            voxel_queue.put((x, y+1))
            voxel_queue.put((x, y-1))
    
    flags[:, :centy] = np.flip(np.flip(flags, axis=0), axis=1)[:, :centy]
    if return_mask:
        return np.fft.ifftshift(flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright]).mean()
    else:
        return np.fft.ifftshift(arr * flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright]).mean()


def _disect_floodfill_3D(arr, cutoff=0.1, radius_cutoff_fraction=0.33, return_mask=False):
    """
    The user *should* input the covariance function, not the autocorrelation

    This will remove the center of the covariance.

    We assume that the arr is input centered (no np.fft.fftshift is necessary)
    
    Array is output centered, along with the mask and the longest direction

    We can also use the space inversion symmetry of the covariance to half the 
    number of computations. 

    :param arr:
    :param cutoff:
    :return:
    """
    
    arr_max = arr.max() * cutoff
    size = min(arr.shape)
    radius_cutoff = radius_cutoff_fraction * size
    xmax = arr.shape[0]
    ymax = arr.shape[1]
    zmax = arr.shape[1]

    centx = int(xmax / 2)
    centy = int(ymax / 2)
    centz = int(zmax / 2)

    flags = np.zeros_like(arr)
    flags[centx, centy, centz] = 1
    voxel_queue = Queue()
    [voxel_queue.put(item) for item in
     [(centx - 1, centy, centz), (centx, centy + 1, centz), (centx + 1, centy, centz), \
             (centx, centy, centz-1), (centx, centy, centz+1)]]

    maxup = 0
    maxdown = 0
    maxleft = 0
    maxupright = 0
    maxdownright = 0

    maxzup = 0
    maxzupright = 0
    maxzupup = 0
    maxzupdown = 0

    maxzdown = 0
    maxzdownright = 0
    maxzdownup = 0
    maxzdowndown = 0


    while not voxel_queue.empty():
        x, y, z = voxel_queue.get()
        direc = np.linalg.norm(np.array([x, y, z]) - np.array([centx, centy, centz]))

        if arr[x, y, z] > arr_max and direc < radius_cutoff and flags[x, y, z] != 1 and x>-1 and y >= centy and x < xmax and y < ymax and z >-1 and z < zmax:

            # These conditions are checking the width of the kernel
            if (x == centx) and (z == centz):
                maxleft = max(maxleft, direc)
            elif y == centy and x < centx and (z==centz):
                maxdown = max(maxdown, direc)
            elif y == centy and x > centx and z==centz:
                maxup = max(maxup, direc)
            elif y == x and z==centz:
                maxupright = max(maxupright, direc)
            elif abs(y - centy) == abs(x - centx) and z==centz:
                maxdownright = max(maxdownright, direc)
            # I started adding here: z>centz
            elif (x==centx) and y==centy and z>centz:
                maxzup = max(maxzup, direc)
            elif (x==centx) and z==y:
                maxzupright = max(maxzupright, direc)
            elif (y==centy) and z==x and z>centz:
                maxzupup = max(maxzupup, direc)
            elif (y==centy) and abs(z-centz)==abs(x-centx) and z>centz:
                maxzupdown = max(maxzupdown, direc)
            # I continued adding: z<centz
            elif (x==centx) and y==centy and z<centz:
                maxzdown = max(maxzdown, direc)
            elif (x==centx) and abs(z-centz)==abs(y-centy) and z<centz:
                maxzdownright = max(maxzdownright, direc)
            elif (y==centy) and abs(z-centz)==abs(x-centx) and z<centz and x>centx:
                maxzdownup = max(maxzdownup, direc)
            elif (y==centy) and z==x and z<centz:
                maxzdowndown = max(maxzdowndown, direc)
            
            flags[x, y, z] = 1.0
            voxel_queue.put((x+1, y, z))
            voxel_queue.put((x-1, y, z))
            voxel_queue.put((x, y+1, z))
            voxel_queue.put((x, y-1, z))
            voxel_queue.put((x, y, z-1))
            voxel_queue.put((x, y, z+1))
    
    flags[:, :centy, :] = np.flip(np.flip(np.flip(flags, axis=0), axis=2), axis=1)[:, :centy, :]
    if return_mask:
        return np.fft.ifftshift(flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright,\
                maxzup, maxzupright, maxzupup, maxzupdown, maxzdown, maxzdownright, \
                maxzdownup, maxzdowndown]).mean()
    else:
        return np.fft.ifftshift(arr * flags), np.array([maxup, maxdown, maxleft, maxupright, maxdownright, \
                maxzup, maxzupright, maxzupup, maxzupdown, maxzdown, maxzdownright, \
                maxzdownup, maxzdowndown]).mean()


def disect(arr1, cutoff=0.2, radius_cutoff_fraction=0.33, twoD=True):
    '''
    A method that wraps the disect function above to transfer from autocorrelation to covariance

    Assumes input autocorrelation is uncentered
    '''
    arr = np.fft.fftn(arr1)
    arr[0,0] = 0.0
    arr = np.fft.ifftn(arr).real
    if twoD:
        cent, max_dir = _disect_floodfill(np.fft.fftshift(arr), cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction, return_mask=True)
        return arr1 * cent, max_dir
    else:
        cent, max_dir = _disect_floodfill_3D(np.fft.fftshift(arr), cutoff=cutoff, radius_cutoff_fraction=radius_cutoff_fraction, return_mask=True)
        return arr1 * cent, max_dir


def rescale(arr, length, desired_length, twoD=True):
    """
    This method and all its called methods assume that the field arr is not centered. 
    """
    if twoD:
        if length > desired_length:
            return downsample(arr, length, desired_length)
        elif length < desired_length:
            return upsample(arr, length, desired_length)
        else:
            return arr
    else:
        if length > desired_length:
            return downsample_3D(arr, length, desired_length)
        elif length < desired_length:
            return upsample_3D(arr, length, desired_length)
        else:
            return arr


def upsample(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    assert desired_length <= np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * length / desired_length
    new_arr = np.zeros_like(arr, dtype=np.complex)
    arr = np.fft.fftshift(arr)[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)]
    arr = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)] = arr
    new_arr = np.fft.ifftn(np.fft.ifftshift(new_arr)).real
    return new_arr

def upsample_3D(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    #
    # This is the 3D implementation of upsample
    assert desired_length <= np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * length / desired_length
    new_arr = np.zeros_like(arr, dtype=np.complex)
    arr = np.fft.fftshift(arr)[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
                            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
                            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)]
    arr = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)] = arr
    new_arr = np.fft.ifftn(np.fft.ifftshift(new_arr)).real
    return new_arr

def downsample(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    assert length < np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * desired_length / length
    new_arr = np.zeros_like(arr)
    arr = np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(arr))[
          int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1),
          int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)])
   
    arr = np.fft.fftshift(np.fft.ifftn(arr).real)
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1)] = arr
    return np.fft.ifftshift(new_arr)

def downsample_3D(arr, length, desired_length):
    # Need to be careful with this. They are incorrectly scaled because of how the FFT is
    # implemented in Numpy. For my application, it doesn't matter. But, for the future it may
    #
    # This is the 3D code. 
    assert length < np.array(arr.shape).min() / 2
    old_scales = arr.shape
    new_scales = np.array(arr.shape) * desired_length / length
    new_arr = np.zeros_like(arr)
    arr = np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(arr))[
          int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1),
          int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1),
          int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)])
   
    arr = np.fft.fftshift(np.fft.ifftn(arr).real)
    new_arr[int(old_scales[0] / 2 - new_scales[0] / 2):int(old_scales[0] / 2 + new_scales[0] / 2 + 1), \
            int(old_scales[1] / 2 - new_scales[1] / 2):int(old_scales[1] / 2 + new_scales[1] / 2 + 1), \
            int(old_scales[2] / 2 - new_scales[2] / 2):int(old_scales[2] / 2 + new_scales[2] / 2 + 1)] = arr
    return np.fft.ifftshift(new_arr)

# ------------------------------------------------------------------------------
# New code that I am working on
# ------------------------------------------------------------------------------

def cross():
    """
    This is a method that returns first the shifts and second the axis for the local
    mean method that is defined below. 
    """
    return [1, -1, 1, -1], [0, 0, 1, 1]

def local_mean(arr, selector=cross):
    """
    This is a method that sums the vector values across all of the surrounding neighbors. Which 
    neighbors is defined by the selector method. 

    We assume that the center voxel is always desired. 
    """
    shifts, axes = selector()
    final_arr = arr.copy()
    for i in range(len(shifts)):
        final_arr += np.roll(arr, shift=shifts[i], axis=axes[i])
    return final_arr

def local_square_mean(arr):
    """
    The above mean method wasn't as general as I had originally hoped. 
    """
    shifts = [(0,1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    final_arr = arr.copy()
    for i in range(len(shifts)):
        final_arr += np.roll(np.roll(arr, shift=shifts[i][0], axis=0), shift=shifts[i][1], axis=1)
    return final_arr

