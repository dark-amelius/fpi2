import cv2
import cython
import math
import sys
import numpy as np
from inspect import getmembers, isfunction
from tqdm import tqdm
from scipy.fftpack import dct, idct
import helpers
    

def dsample_slow(amount, image):
    h = image.shape[0]
    w = image.shape[1]


    for y in tqdm(range(0,h)):
        for x in range(0, w):
            image[y, x, 1] = amount * round(image[y,x,1]/amount)
            image[y, x, 2] = amount * round(image[y,x,2]/amount)
    return image

def jpg_compress(file_name):
    image = cv2.imread(file_name)
    
    # Convert Color to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Downsample
    imageShift = helpers.level_shift(image)

    # Generate new image
    h, w = image.shape
    pad_h = 8 - (h % 8)
    pad_w = 8 - (w % 8)
    if pad_h == 8: pad_h = 0
    if pad_w == 8: pad_w = 0
    n_image = np.zeros((h+pad_h, w+pad_w))
    n_image[:h, :w] = imageShift[:h, :w]

    print(n_image.shape)
    # Apply dct
    dct = helpers.compress(n_image, 50)
    decomp = helpers.decompress(dct, 50)
    cv2.imshow("decomp", decomp)
    cv2.waitKey(0)

    psnr = helpers.psnr(image, decomp)
    print(psnr)
    


jpg_compress("./test.jpg")