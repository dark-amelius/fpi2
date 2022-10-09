import cv2
import numpy as np
import helpers
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def squared_distance(v, w):
    return np.linalg.norm(v-w)

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_sum(vectors):
    num_elements = len(vectors[0])
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


class KMeans:
    def __init__(self, k):
        self.k = k          
        self.means = None   

    def classify(self, input):
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs, iters):

        self.means = random.sample(inputs, self.k)
        classes = None

        for j in tqdm(range(iters)):
            
            n_classes = map(self.classify, inputs)
            
            if n_classes == classes:
                return
            
            classes = n_classes
            i_points = [[] for _ in range(self.k)]
            l_classes = list(classes)
            for i in range(len(inputs)):
                i_points[l_classes[i]].append(inputs[i])
            for i in range(self.k):
                if i_points[i]:
                    self.means[i] = vector_mean(i_points[i])
            
def recolor(pixel, clusterer):
    cluster = clusterer.classify(pixel)        # index of the closest cluster
    val = clusterer.means[cluster]
    val[0] = int(val[0])
    val[1] = int(val[1])
    val[2] = int(val[2])
    return val


def k_cluster(k, image_path):
    img = mpimg.imread(image_path)
    img = img.astype(np.int64)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(5)
    clusterer.train(pixels, 100)   # this might take a while
    new_img = [[recolor(pixel, clusterer) for pixel in row]   # recolor this row of pixels
           for row in img]
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    

k_cluster(5, './test_helltaker.jpg')