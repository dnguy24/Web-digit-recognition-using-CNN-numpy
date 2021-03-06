import numpy as np
from matplotlib import pyplot as plt
from CNN.model import Model
from scipy import ndimage
from PIL import Image
import random

def crop_img(img_ndarray):
    """
    Crop white space around digit.
    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (determined by canvas size)
        Image to crop (drawn by user).
    Returns
    -------
    Cropped image as 2D numpy ndarray
    """
    # Length of zero pixel values for rows and columns
    # Across rows
    first_row = np.nonzero(img_ndarray)[0].min()
    last_row = np.nonzero(img_ndarray)[0].max()
    middle_row = np.mean([last_row, first_row])
    # Across cols
    first_col = np.nonzero(img_ndarray)[1].min()
    last_col = np.nonzero(img_ndarray)[1].max()
    middle_col = np.mean([last_col, first_col])

    # Crop by longest non-zero to make sure all is kept
    row_length = last_row - first_row
    col_length = last_col - first_col
    length = max(row_length, col_length)
    # Minimum size of 28x28
    length = max(length, 28)

    # Get half length to add to middle point (add some padding: 1px)
    half_length = (length / 2) + 1

    # Make sure even the shorter dimension is centered
    first_row = int(middle_row - half_length)
    last_row = int(middle_row + half_length)
    first_col = int(middle_col - half_length)
    last_col = int(middle_col + half_length)

    # Crop image
    return img_ndarray[first_row:last_row, first_col:last_col]


def center_img(img_ndarray):
    """
    Center digit on center of mass of the pixels.
    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (determined by digit size)
        Image to center (drawn by user).
    Returns
    -------
    Centered image as 2D numpy ndarray
    """
    # Compute center of mass and frame center
    com = ndimage.measurements.center_of_mass(img_ndarray)
    center = len(img_ndarray) / 2

    # Center by adding lines of zeros matching diff between
    # center of mass and frame center
    row_diff = int(com[0] - center)
    col_diff = int(com[1] - center)

    rows = np.zeros((abs(row_diff), img_ndarray.shape[1]))
    if row_diff > 0:
        img_ndarray = np.vstack((img_ndarray, rows))
    elif row_diff < 0:
        img_ndarray = np.vstack((rows, img_ndarray))

    cols = np.zeros((img_ndarray.shape[0], abs(col_diff)))
    if col_diff > 0:
        img_ndarray = np.hstack((img_ndarray, cols))
    elif col_diff < 0:
        img_ndarray = np.hstack((cols, img_ndarray))

    # Make image square again (add zero rows to the smaller dimension)
    dim_diff = img_ndarray.shape[0] - img_ndarray.shape[1]
    half_A = half_B = abs(int(dim_diff / 2))

    if dim_diff % 2 != 0:
        half_B += 1

    # Add half to each side (to keep center of mass)
    # Handle dim_diff == 1
    if half_A == 0:  # 1 line off from exactly centered
        if dim_diff > 0:
            half_B = np.zeros((img_ndarray.shape[0], half_B))
            img_ndarray = np.hstack((half_B, img_ndarray))
        else:
            half_B = np.zeros((half_B, img_ndarray.shape[1]))
            img_ndarray = np.vstack((half_B, img_ndarray))

    elif dim_diff > 0:
        half_A = np.zeros((img_ndarray.shape[0], half_A))
        half_B = np.zeros((img_ndarray.shape[0], half_B))
        img_ndarray = np.hstack((img_ndarray, half_A))
        img_ndarray = np.hstack((half_B, img_ndarray))
    else:
        half_A = np.zeros((half_A, img_ndarray.shape[1]))
        half_B = np.zeros((half_B, img_ndarray.shape[1]))
        img_ndarray = np.vstack((img_ndarray, half_A))
        img_ndarray = np.vstack((half_B, img_ndarray))
    # Add padding all around (15px of zeros)
    return np.lib.pad(img_ndarray, 15, 'constant', constant_values=(0))


def resize_img(img_ndarray):
    """
    Resize digit image to 28x28.
    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (depending on drawing)
        Image to resize.
    Returns
    -------
    Resized image as 2D numpy ndarray (28x28)
    """
    img = Image.fromarray(img_ndarray)
    img.thumbnail((28, 28), Image.ANTIALIAS)

    return np.array(img)

def printarray(array):
    nparray = np.asarray(array).reshape(280, 280, 4)
    # print(np.max(nparray))
    # print(np.count_nonzero(nparray))
    pixels = np.array(np.max(nparray, axis=2))
    print(pixels.shape)
    plt.imsave(arr=pixels, cmap = 'gray', fname='pic1.png')
    pixels = crop_img(pixels)
    plt.imsave(arr=pixels, cmap = 'gray', fname='pic2.png')
    pixels = center_img(pixels)
    # pixels = center_img(pixels)
    # print(pixels.shape)
    # plt.imsave(arr=pixels, cmap = 'gray', fname='pic0.png')
    # temp = np.zeros((28, 28))
    # for y in range(28):
    #     for x in range(28):
    #         mean = 0
    #         for v in range(10):
    #             for h in range(10):
    #                 mean+=pixels[y*10+v][x*10+h]
    #         mean = np.mean(mean)
    #         temp[y][x] = mean
    # input_size = 280
    # output_size = 28
    # bin_size = input_size // output_size
    # small_image =
    # plt.imshow(pixels, cmap='gray')
    temp = resize_img(pixels)
    print(temp.shape)
    plt.imsave(arr=temp, cmap = 'gray', fname='pic.png')
    model = Model()
    pred = model.classify(temp)
    print(pred)
    return str(pred)
    # print(nparray)