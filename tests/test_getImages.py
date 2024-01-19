import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from Projekt.src.getImages import ImageHandler


if __name__ == "__main__":
    images = ImageHandler()
    images.show_info()
    imgs = images.get_paths_from_folder("Normal")
    print(imgs)
    nb_tests = 2
    for i in range(nb_tests):
        img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Image", img)
        cv2.waitKey()
