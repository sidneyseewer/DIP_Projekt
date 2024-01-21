import getImages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import python.imutils as imutils
import python.initdata as init
import cvHelper
import glob

"""
subfolders = [
    '0-Normal', '1-NoHat', '2-NoFace', '3-NoLeg', '4-NoBodyPrint',
    '5-NoHand', '6-NoHead', '7-NoArm', 'All', 'Combinations', 'Other', 'templates'
]
"""

class DebugLevel(Enum):
    DEBUG = 0
    INFO = 1
    PRODUCTION = 2

def retrieve_images(subFolder: str = None):
        imgHandler = getImages.ImageHandler()
#        imgHandler.show_info()
        if subFolder is None:
            return imgHandler.get_all_images()
        else:
             for subFolderNames in imgHandler.subfolders:
                  if subFolder in subFolderNames:
                       subFolder = subFolderNames
                       print(f"Chose folder {subFolder}")
                       break
             return imgHandler.get_paths_from_folder(subFolder)

def subtractBackground(debug_level = DebugLevel.PRODUCTION, image = None, background_image = None):
    # Subtract Background
    img_no_background = imutils.shadding(image, background_image)

    return img_no_background

def convertToBinary(debug_level = DebugLevel.PRODUCTION, img_no_background = None):
    im_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    (thresh, imgBW) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    return imgBW

def clearBorder(debug_level = DebugLevel.PRODUCTION, imgBW = None):
    imclearborder = imutils.imclearborder(imgBW, 15)

    return imclearborder

# def opening(debug_level = DebugLevel.PRODUCTION, image = None):
#     im_opened = imutils.bwareaopen(image, 200)
    
#     return im_opened

def opening(debug_level = DebugLevel.PRODUCTION, image = None, SE_size = 20):
    # apply closing operation using a SE
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (SE_size, SE_size))
    # borderType is constant by default
    closing = cv2.morphologyEx(image, cv2.MORPH_OPEN, se, iterations=1)

    return closing

def closing(debug_level = DebugLevel.PRODUCTION, image = None, SE_size = 20):
    # apply closing operation using a SE
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (SE_size, SE_size))
    # borderType is constant by default
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se, iterations=1)

    return closing

def correct90DegreeRectangle(rect = None):
    """
    Swaps width and height and corrects the Rectangle if width > height
    """
    width = rect[1][0]
    height = rect[1][1]
    if(width > height):
        rect = ((rect[0][0], rect[0][1]), (height, width), rect[2] + 90)
    return rect

def drawRectanle(image = None, rect = 0):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rectangle on the original image/copy
    image_with_rect = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)
    # cv2.ellipse(image_with_rect,rect, ell_rot, (128,128,0),2)

    return image_with_rect

def pipeline(debug_level = DebugLevel.PRODUCTION, img = None):
    print("PIPELINE STARTED")
    imgbackground = cv2.imread('./img/Other/image_100.jpg')

    print("PREPROCESS IMAGE")
    img_no_background = subtractBackground(debug_level, img, imgbackground)
    cv2.imshow("img_no_background",img_no_background)
    img_no_background_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_no_background_gray",img_no_background_gray)
    img_no_background_binary = convertToBinary(debug_level, img_no_background)
    cv2.imshow("img_no_background_binary",img_no_background_binary)
    img_no_background_binary_no_border = clearBorder(debug_level, img_no_background_binary)
    cv2.imshow("img_no_background_binary_no_border",img_no_background_binary_no_border)
    img_no_background_binary_no_border_opened = opening(debug_level, img_no_background_binary_no_border, SE_size=20)
    cv2.imshow("img_no_background_binary_no_border_opened",img_no_background_binary_no_border_opened)
    # img_before_no_background_binary_no_border = clearBorder(debug_level, img_no_background_binary_no_border_opened)
    # cv2.imshow("img_before_no_background_binary_no_border",img_before_no_background_binary_no_border)
    img_no_background_binary_no_border_opened_closed = closing(debug_level= DebugLevel.PRODUCTION, image=img_no_background_binary_no_border_opened, SE_size=20)
    cv2.imshow("img_no_background_binary_no_border_opened_closed",img_no_background_binary_no_border_opened_closed)

    

    count_non_zero = cv2.countNonZero(img_no_background_binary_no_border)
    if(count_non_zero <100):
        print("IMAGE EMPTY")
    else:
        print("IMAGE OKAY")
        print("LOCATE RECTANGLE")
        img_props = imutils.regionprops(img_no_background_binary_no_border)
        contours, area_vec, [cx, cy], rect, ell_rot = img_props
        
        print("CORRECT RECTANGLE")
        rect = correct90DegreeRectangle(rect)
        img_no_background_gray_rectangle = drawRectanle(image=img_no_background_gray, rect=rect)

        cv2.imshow("img_no_background_gray_rectangle",img_no_background_gray_rectangle)

    cv2.imshow("img",img)
    print("\n")
    # cv2.waitKey()

    
    print("ROTATE IMAGE")


if __name__ == "__main__":
    imageDir = "./img/All/"

    for imagePath in glob.glob(imageDir + "*.jpg"):
        print("IMAGE: " +str(imagePath) + "\n")
        img = cv2.imread(imagePath)
        pipeline(DebugLevel.PRODUCTION, img = img)


# "C:\development\Hagenberg\DIP\DIP_Projekt\img\All\image_100.jpg"
    
    # imageDir = "./img/All/image_100.jpg"
    # img = cv2.imread(imageDir)
    # pipeline(DebugLevel.PRODUCTION, img = img)