import numpy as np
import matplotlib.pyplot as plt
import cv2

# Image utilies
def load_image(path):
    img = cv2.imread(path)
    return img

def save_image(img, file):
    cv2.imwrite('Output/{file}.png'.format(file=file),img,)

def deprocess_image(img):
    img = img / img.max()
    img = 255 * img
    img = img.astype(np.uint8)
    return  img

def rotate_image(img, clockwise):
    return cv2.rotate(img,clockwise)

def display_image(img):
    img = deprocess_image(img)
    plt.imshow(img)
    plt.show()