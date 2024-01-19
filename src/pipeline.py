import getImages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import python.imutils as imutils

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

def get_background_image():
    imgs = retrieve_images("Other")
    print(imgs)
    for img in imgs:
        if "image_100" in img:
            return img
    return None

def showImage(img, debug_level = DebugLevel.DEBUG, name = "Image"):
    if img is None:
        return
    if debug_level.value < 2:
        cv2.imshow(name,img)
        cv2.waitKey()

def subImages(main_image, image_to_subtract):
    # Check if images are loaded successfully
    if main_image is None or image_to_subtract is None:
        print("Error: Could not load images.")
        return None

    # Ensure both images have the same dimensions
    if main_image.shape != image_to_subtract.shape:
        print(f"Error: Shape of images to substract doesn't match: {main_image.shape}, {image_to_subtract.shape}")
        return None

    # Calculate absolute difference
    difference = cv2.absdiff(main_image, image_to_subtract)

    return difference

def threshhold_image(image, block_size=11,c_value=2):
    # Apply a binary threshold to the subtracted image
    thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, c_value)

    return thresholded_image

def removeSaltPeper(binary_image, min_area_threshold=1000):
    # Create a structuring element for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological closing (dilation followed by erosion)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the large enclosed shapes
    mask = np.zeros_like(binary_image)

    # Iterate through contours and fill the large enclosed shapes in the mask
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Bitwise AND to keep only the large enclosed shapes in the original image
    result_image = cv2.bitwise_and(binary_image, mask)

    return result_image

def start_pipeline(debug_level = DebugLevel.DEBUG, img = None):
    # Get background
    background_image_path = get_background_image()
    if background_image_path is None:
        print("Could not geet Background Image")
        return
    print("Backgournd Image found")
    background_image = cv2.imread(background_image_path)
    showImage(background_image,debug_level=debug_level, name="Background Image")
    # Get image
    if img is None:
        normal_imgs = retrieve_images("Normal")
        img = normal_imgs[1]
    image = cv2.imread(img)
    showImage(image,debug_level=debug_level,name="Image to treat")
    
    # Subtract Background
    img_no_background = imutils.shadding(image, background_image)
    showImage(img_no_background, debug_level=debug_level,name="Subtracted Background")

    im_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    (thresh, imgBW) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    showImage(imgBW, debug_level=debug_level,name="BW")

    imclearborder = imutils.imclearborder(imgBW, 15)
    showImage(imclearborder, debug_level=debug_level,name="Cleared the Border")

    im_opened = imutils.bwareaopen(imclearborder, 200)
    showImage(im_opened, debug_level=debug_level,name="Opened")

    # extract rectangle properties
    img_props = imutils.regionprops(im_opened)
    contours, area_vec, [cx, cy], rect, ell_rot = img_props
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rectangle on the original image/copy
    image_with_rect = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)

    showImage(image_with_rect, debug_level=debug_level,name="Rectangle")

if __name__ == "__main__":
    start_pipeline(DebugLevel.DEBUG)