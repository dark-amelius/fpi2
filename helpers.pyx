import numpy as np
cimport numpy as np
import cv2
import cython
import math
import random
from tqdm import tqdm

np.import_array()

@cython.boundscheck(False)
cpdef np.ndarray level_shift(np.ndarray image):
    cdef int x, y, w, h
    image = image.copy().astype(np.int)
    x = 0
    y = 0
    h = image.shape[0]
    w = image.shape[1]

    for y in tqdm(range(0,h)):
        for x in range(0, w):
            image[y, x] = image[y,x] - 128
    return image.astype(np.int)

@cython.boundscheck(False)
cpdef np.ndarray compress(np.ndarray image, int quality_level):
    image = image.copy().astype(np.float32)
    cdef np.ndarray T = np.array([
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975,-0.0975,-0.2778,-0.4157,-0.4904],
        [0.4619, 0.1913,-0.1913,-0.4619,-0.4619,-0.1913, 0.1913, 0.4619],
        [0.4157,-0.0975,-0.4904,-0.2778, 0.2778, 0.4904, 0.0975,-0.4157],
        [0.3536,-0.3536,-0.3536, 0.3536, 0.3536,-0.3536,-0.3536, 0.3536],
        [0.2778,-0.4904, 0.0975, 0.4157,-0.4157,-0.0975, 0.4904,-0.2778],
        [0.1913,-0.4619, 0.4619,-0.1913,-0.1913, 0.4619,-0.4619, 0.1913],
        [0.0975,-0.2778, 0.4157,-0.4904, 0.4904,-0.4157, 0.2778,-0.0975]])

    cdef np.ndarray T_t = T.transpose()

    cdef np.ndarray Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    if(quality_level > 50):
        Q = Q * ((100-quality_level)/50.0)

    elif (quality_level < 50):
        Q = Q * (50.0/quality_level)


    Q = 1/Q
    cdef int x, y, w, h
    h = image.shape[0]
    w = image.shape[1]

    for y in tqdm(range(0, h-7, 8)):
        for x in range(0, w-7, 8):
            image[y:y+8, x:x+8] = np.matmul(T, image[y:y+8, x:x+8])
            image[y:y+8, x:x+8] = np.matmul(image[y:y+8, x:x+8], T_t)
            image[y:y+8, x:x+8] = np.matrix.round(np.multiply(image[y:y+8, x:x+8], Q))

    return image.astype(np.int32)


    
cpdef np.ndarray decompress(np.ndarray image, int quality_level):
    image = image.copy().astype(np.float32)
    
    cdef np.ndarray T = np.array([
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975,-0.0975,-0.2778,-0.4157,-0.4904],
        [0.4619, 0.1913,-0.1913,-0.4619,-0.4619,-0.1913, 0.1913, 0.4619],
        [0.4157,-0.0975,-0.4904,-0.2778, 0.2778, 0.4904, 0.0975,-0.4157],
        [0.3536,-0.3536,-0.3536, 0.3536, 0.3536,-0.3536,-0.3536, 0.3536],
        [0.2778,-0.4904, 0.0975, 0.4157,-0.4157,-0.0975, 0.4904,-0.2778],
        [0.1913,-0.4619, 0.4619,-0.1913,-0.1913, 0.4619,-0.4619, 0.1913],
        [0.0975,-0.2778, 0.4157,-0.4904, 0.4904,-0.4157, 0.2778,-0.0975]])

    cdef np.ndarray T_t = T.transpose()

    cdef np.ndarray Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    if(quality_level > 50):
        Q = Q * ((100-quality_level)/50.0)

    elif (quality_level < 50):
        Q = Q * (50.0/quality_level)

    cdef int x, y, w, h
    h = image.shape[0]
    w = image.shape[1]

    for y in tqdm(range(0, h-7, 8)):
        for x in range(0, w-7, 8):
            image[y:y+8, x:x+8] = np.multiply(image[y:y+8, x:x+8], Q)
            image[y:y+8, x:x+8] = np.matmul(T_t, image[y:y+8, x:x+8])
            image[y:y+8, x:x+8] = np.matmul(image[y:y+8, x:x+8], T)
            image[y:y+8, x:x+8] = np.matrix.round(image[y:y+8, x:x+8]) + 128
    
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype(np.uint8)



cpdef double psnr(np.ndarray reference, np.ndarray test):
    cdef int h, w, x, y

    h = reference.shape[0]
    w = reference.shape[1]


    reference = reference.astype(np.float64)
    test = test.astype(np.float64)

    cdef double term1 = (np.max(reference) ** 2)
    cdef double sum1 = 0

    for y in tqdm(range(h)):
        for x in range(w):
            sum1 += ((reference[y][x] - test[y][x]) ** 2)
    sum1 = sum1/(h*w)
    return 10 * np.log10(term1/sum1)

@cython.boundscheck(False)
cpdef list point_loop(list inputs, list assignments, list means, int k):
    cdef list i_points = [[] for _ in range(k)]
    cdef int i
    for i in range(len(inputs)):
        i_points[assignments[i]].append(inputs[i])
    for i in range(k):
        if i_points[i]:
            means[i] = np.mean(i_points[i])
    return means
