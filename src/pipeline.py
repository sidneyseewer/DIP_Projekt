import getImages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import python.imutils as imutils
import python.initdata as init

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
        if(img.shape[1] < 200):
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_AREA)
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

def pad_to_center(image1, image2):
    """
    Pad the smaller image among image1 and image2 to match the size of the larger one,
    keeping the original image centered.

    Parameters:
    image1 (numpy.ndarray): First input image.
    image2 (numpy.ndarray): Second input image.

    Returns:
    tuple: A tuple containing the possibly padded images (image1, image2).
    """

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Determine padding for height and width for each image
    pad_h1 = max(h2 - h1, 0)
    pad_w1 = max(w2 - w1, 0)
    pad_h2 = max(h1 - h2, 0)
    pad_w2 = max(w1 - w2, 0)

    # Distribute padding evenly on both sides
    pad_top1, pad_bot1 = pad_h1 // 2, pad_h1 - pad_h1 // 2
    pad_left1, pad_right1 = pad_w1 // 2, pad_w1 - pad_w1 // 2
    pad_top2, pad_bot2 = pad_h2 // 2, pad_h2 - pad_h2 // 2
    pad_left2, pad_right2 = pad_w2 // 2, pad_w2 - pad_w2 // 2

    # Pad the images accordingly
    if pad_h1 > 0 or pad_w1 > 0:
        image1 = cv2.copyMakeBorder(image1, pad_top1, pad_bot1, pad_left1, pad_right1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if pad_h2 > 0 or pad_w2 > 0:
        image2 = cv2.copyMakeBorder(image2, pad_top2, pad_bot2, pad_left2, pad_right2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image1, image2
def subtractBackground(debug_level = DebugLevel.DEBUG, image = None, background_image = None):
    # Subtract Background
    img_no_background = imutils.shadding(image, background_image)
    showImage(img_no_background, debug_level=debug_level,name="Subtracted Background")
    
    return img_no_background

def convertToBW(debug_level = DebugLevel.DEBUG, img_no_background = None):
    im_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    (thresh, imgBW) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    showImage(imgBW, debug_level=debug_level,name="BW")
    
    return imgBW

def clearBorder(debug_level = DebugLevel.DEBUG, imgBW = None):
    imclearborder = imutils.imclearborder(imgBW, 15)
    showImage(imclearborder, debug_level=debug_level,name="Cleared the Border")

    return imclearborder

def opening(debug_level = DebugLevel.DEBUG, imclearborder = None):
    im_opened = imutils.bwareaopen(imclearborder, 200)
    showImage(im_opened, debug_level=debug_level,name="Opened")
    
    return im_opened

def closing(debug_level = DebugLevel.DEBUG, im_opened = None, SE_size = 20):
    # apply closing operation using a SE
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # borderType is constant by default
    closing = cv2.morphologyEx(im_opened, cv2.MORPH_CLOSE, se, iterations=1)
    showImage(closing, debug_level=debug_level,name="Img closing")

    return closing

def drawRectanle(debug_level = DebugLevel.DEBUG, image = None, rect = 0):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rectangle on the original image/copy
    image_with_rect = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)
    # cv2.ellipse(image_with_rect,rect, ell_rot, (128,128,0),2)
    showImage(image_with_rect, debug_level=debug_level,name="Rectangle")

    return image_with_rect

def getCorrectRotationAngle(debug_level = DebugLevel.DEBUG, im_closed = None, cx = 0, cy = 0, rect = 0):
    template, defects = init.initdata()
    perfect_angle = 0
    biggest_count_non_zeros = 0
    angle_offset= [0, 180]

    showImage(template["img_mask"], debug_level=DebugLevel.DEBUG,name="template")

    for offset in angle_offset:
        angle= rect[2] + offset
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Perform the rotation
        rotated_image_bw = cv2.warpAffine(im_closed, M, im_closed.shape[1::-1])

        # Get the size of the rotated rectangle
        size = (int(rect[1][0]), int(rect[1][1]))

        # Extract the straightened ROI
        x, y = np.int0((cx, cy))
        x -= size[0] // 2
        y -= size[1] // 2
        straightened_roi_bw = rotated_image_bw[y:y+size[1], x:x+size[0]]

        straightened_roi_bw, template["img_mask"] = pad_to_center(straightened_roi_bw, template["img_mask"])
        
        bitwise_and = cv2.bitwise_and(straightened_roi_bw, template["img_mask"])

        
        showImage(straightened_roi_bw, debug_level=DebugLevel.DEBUG,name="Found")
        showImage(bitwise_and, debug_level=DebugLevel.DEBUG,name = str(offset) + " degree subtracted template ")

        count_non_zeros = cv2.countNonZero(bitwise_and)
        
        if(count_non_zeros >= biggest_count_non_zeros):
            biggest_count_non_zeros = count_non_zeros
            perfect_angle = angle

    return perfect_angle


def rotateImage(debug_level = DebugLevel.DEBUG, image = None, angle = 0, center = (0, 0)):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, image.shape[1::-1])
    showImage(rotated, debug_level=debug_level,name="Rotated image")

    return rotated

def correctRectangle(rect = None):
    """
    Swaps width and height and corrects the Rectangle if width > height
    """
    width = rect[1][0]
    height = rect[1][1]
    if(width > height):
        rect = ((rect[0][0], rect[0][1]), (height, width), rect[2] + 90)
    return rect

def getTemplatePart(path = None):
    """
    Extracts a part of the template.
    Param: path to the mask of the needed part. 
    """
    template, defects = init.initdata()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return cv2.bitwise_and(mask,template['imgg'])

def start_pipeline(debug_level = DebugLevel.DEBUG, img = None):

    # extract part of template
    template, defects = init.initdata()
    hat_template = getTemplatePart("./img/templates/mask_hat.png")
    cv2.imshow("img0",template['imgg'])
    cv2.imshow("hat_template",hat_template)
    
    cv2.waitKey()

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
        normal_imgs = retrieve_images("All")
        img = normal_imgs[9]
    image = cv2.imread(img)

    showImage(image,debug_level=debug_level,name="Image to treat")
    
    img_no_background = subtractBackground(debug_level, image, background_image)
    imgBW = convertToBW(debug_level, img_no_background)
    imclearborder = clearBorder(debug_level, imgBW)
    im_opened = opening(debug_level, imclearborder)
    im_closed = closing(debug_level= DebugLevel.DEBUG, im_opened=im_opened, SE_size=20)

    # # extract rectangle properties
    # img_props = imutils.regionprops(im_closed)
    # contours, area_vec, [cx, cy], rect, ell_rot = img_props
    
    # rect = correctRectangle(rect)

    # image_with_rect = drawRectanle(DebugLevel.DEBUG, image=image, rect=rect)
    
    # print("width= ", rect[1][0])

    # angle = getCorrectRotationAngle(debug_level, im_closed, cx, cy, rect)

    # print("Rect angle ", rect[2])
    # print("corrected angle ", angle)
    # img_rotated = rotateImage(debug_level= DebugLevel.DEBUG, image= image, angle=rect[2], center = (cx, cy))
if __name__ == "__main__":
    all_imgs = retrieve_images("Normal")

    for img in all_imgs:
        start_pipeline(DebugLevel.DEBUG, img = img)
    
    # start_pipeline(DebugLevel.PRODUCTION, img ="./img/0-Normal/image_017.jpg")