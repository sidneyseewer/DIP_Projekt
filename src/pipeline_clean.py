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

def clearBorder(debug_level = DebugLevel.PRODUCTION, imgBW = None, clearBorderRadius = None):
    imclearborder = imutils.imclearborder(imgBW, clearBorderRadius)

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

def draw_ellipse(image, ellipse):
    """
    Draw an ellipse on an image.
    """
    return cv2.ellipse(image, ellipse,(128,128,0),2)

def floodFill(binary_image = None):
    img_floodfill = binary_image.copy()

    # Create a mask that is two pixels bigger than the source image
    h, w = binary_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Specify a seed point - (0, 0) is a common choice for binary images
    seed_point = (0, 0)

    # Perform flood fill
    cv2.floodFill(img_floodfill, mask, seed_point, 255)

    # Invert the flood filled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    
    img_out = binary_image | img_floodfill_inv

    return img_out

def rotate(image = None, cx = 0, cy = 0, angles = 0):
    images = []
    for angle in angles:
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Perform the rotation
        rotated_image_bw = cv2.warpAffine(image, M, image.shape[1::-1])

        images.append(rotated_image_bw)
        
    return images

def getTemplatePart(mask_name = None):
    """
    Extracts a part of the template.
    Param: path to the mask of the needed part. 
    """
    template, defects = init.initdata()
    template_img_gray = cv2.cvtColor(template["img"], cv2.COLOR_BGR2GRAY)
    # mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # dunno if first part is needed 
    return (defects[mask_name]['mask'] * template_img_gray, defects[mask_name]['mask'])

def matchTemplate(image = None, template_image = None, template_mask = None):
    # apply zero-mean cross correlation
    res = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF, mask=template_mask)
    # search for highest match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("Zero-Mean Cross Correlation Coefficients ranges from {} to {}".format(min_val, max_val))

    return max_val

def getBestImage(images, template_part, template_mask):
    max_val = 0
    best_image = None

    for image in images:
        temp_max_val = matchTemplate(image, template_part, template_mask)

        if(temp_max_val > max_val):
            max_val = temp_max_val
            best_image = image

    return (max_val, best_image)



def pipeline(debug_level = DebugLevel.PRODUCTION, img = None):
    print("PIPELINE STARTED")
    imgbackground = cv2.imread('./img/Other/image_100.jpg')
    SE_size = 5
    clearBorderRadius= 10

    print("PREPROCESS IMAGE")
    img_no_background = subtractBackground(debug_level, img, imgbackground)
    cv2.imshow("img_no_background",img_no_background)
    img_no_background_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_no_background_gray",img_no_background_gray)
    img_no_background_binary = convertToBinary(debug_level, img_no_background)
    cv2.imshow("img_no_background_binary",img_no_background_binary)
    img_no_background_binary_floodFill = floodFill(img_no_background_binary)
    cv2.imshow("img_no_background_binary_floodFill",img_no_background_binary_floodFill)
    img_no_background_binary_floodFill_no_border = clearBorder(debug_level, img_no_background_binary_floodFill, clearBorderRadius)
    cv2.imshow("img_no_background_binary_floodFill_no_border",img_no_background_binary_floodFill_no_border)
    img_no_background_binary_floodFill_no_border_closed = closing(debug_level= DebugLevel.PRODUCTION, image=img_no_background_binary_floodFill_no_border, SE_size=SE_size)
    cv2.imshow("img_no_background_binary_floodFill_no_border_opened_closed", img_no_background_binary_floodFill_no_border_closed)
    img_no_background_binary_floodFill_no_border_closed_opened = opening(debug_level, img_no_background_binary_floodFill_no_border_closed, SE_size=SE_size)
    cv2.imshow("img_no_background_binary_floodFill_no_border_opened",img_no_background_binary_floodFill_no_border_closed_opened)
    

    count_non_zero = cv2.countNonZero(img_no_background_binary_floodFill_no_border_closed_opened)
    if(count_non_zero <100):
        print("IMAGE EMPTY")

        return
    

    print("IMAGE OKAY")
    print("LOCATE RECTANGLE")
    img_props = imutils.regionprops(img_no_background_binary_floodFill_no_border_closed_opened)
    contours, area_vec, [cx, cy], rect, ellipse = img_props
    
    # ellipse_rotation = ellipse[2]

    print("CORRECT RECTANGLE")
    rect = correct90DegreeRectangle(rect)
    print(rect)
    img_no_background_gray_rectangle = drawRectanle(image=img_no_background_gray, rect=rect)
    img_no_background_gray_ellipse = draw_ellipse(img_no_background_gray_rectangle, ellipse)
    # cv2.imshow("img_no_background_gray_rectangle", img_no_background_gray_rectangle)
    cv2.imshow("img_no_background_gray_ellipse", img_no_background_gray_ellipse)
    
    # cv2.imshow("img",img)
    # print("\n")
    # cv2.waitKey()

    print("ROTATE IMAGE")
    # also correct if rectangle is upside down
    angles = [rect[2], ellipse[2], rect[2] + 180, ellipse[2] + 180]
    images_corrected_angle = rotate(image = img_no_background_gray, cx = cx, cy = cy, angles = angles)
    
    # show corrected angles
    # cv2.imshow("images_corrected_angle[0]", images_corrected_angle[0])
    # cv2.imshow("images_corrected_angle[1]", images_corrected_angle[1])
    # cv2.imshow("images_corrected_angle[2]", images_corrected_angle[2])
    # cv2.imshow("images_corrected_angle[3]", images_corrected_angle[3])

    print("TEMPLATE MATCHING")
    template_part, template_mask = getTemplatePart("body print")
    cv2.imshow("template_part", template_part)
    (max_val, best_image) = getBestImage(images_corrected_angle, template_part, template_mask)
    
    cv2.imshow("best_image", best_image)

    print("END")


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