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

def rotate(image = None, cx = 0, cy = 0, angles = 0, rect = None):
    images = []
    cropped_images = []
    for angle in angles:
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Perform the rotation
        rotated_image_bw = cv2.warpAffine(image, M, image.shape[1::-1])
        images.append(rotated_image_bw)

        if( rect is not None):
            # Adjust ROI Size to be at least as big as the template
            width, height = (int(rect[1][0]), int(rect[1][1]))
            if width < 124:
                width = 124
            if height < 200:
                height = 200
            size = (width, height)

            # Extract the straightened ROI
            x, y = np.int0((cx, cy))
            x -= size[0] // 2
            y -= size[1] // 2
            straightened_roi_bw = rotated_image_bw[y:y+size[1], x:x+size[0]]
            cropped_images.append(straightened_roi_bw)
        
    return images, cropped_images

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

def getAllTemplateParts():
    """
    Extracts aall parts of the template.
    """
    template, defects = init.initdata()
    template_parts = []
    for class_label, defect_type in enumerate(defects):

        template_img_gray = cv2.cvtColor(template["img"], cv2.COLOR_BGR2GRAY)
        
        template_parts.append((class_label,template_img_gray, defects[defect_type]["mask"]))

    return template_parts

def matchTemplate(image = None, template_image = None, template_mask = None):

    # do not pad template.. if not needed if return values is not used
    # just for error handling, should never be the case
    if(image.shape[0] < template_image.shape[0] or image.shape[1] < template_image.shape[1]):
        image, _ = pad_to_center(image, template_image)

    # apply zero-mean cross correlation
    res = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED, mask=template_mask)
    # search for highest match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("Zero-Mean Cross Correlation Coefficients ranges from {} to {}".format(min_val, max_val))
    # cvHelper.showRectangle(image, max_loc, template_image.shape[0], template_image.shape[1])

    return max_val

def matchTemplateWithVariation(image, template_part, template_mask, variation):
    variation_max_val = 0
    angles = []
    for angle in range(-variation, variation, 1):
        angles.append(angle)

    template_part_variations, _ = rotate(template_part, template_part.shape[0] // 2, template_part.shape[1] // 2, angles)
    template_mask_variations, _ = rotate(template_mask, template_mask.shape[0] // 2, template_mask.shape[1] // 2, angles)

    for i in range(0, 2* variation, 1):
        temp_max_val = matchTemplate(image, template_part_variations[i], template_mask_variations[i])

        if(temp_max_val == float('inf')):
            print("============= WARNING: max value inf??? =============")
        elif(temp_max_val >= variation_max_val):
            variation_max_val = temp_max_val
    
    return variation_max_val

def detectDefects(image, matching_thresh):
    template_parts = getAllTemplateParts()
    detected_parts = []
    detected_defects = []
        
    for template_label, template_img_gray, template_mask in template_parts:
        # cv2.imshow("template_part", template_img_gray)

        temp_max_val = matchTemplateWithVariation(image, template_img_gray, template_mask, 5)
        # temp_max_val = matchTemplate(image, template_part, template_mask)

        if(temp_max_val == float('inf')):
            print("============= WARNING: how to handle max_value inf??? =============")
            # detected_defects.append(template_label, template_img_gray)
        elif(temp_max_val >= matching_thresh):
            # todo: need name here
            detected_parts.append((template_label, temp_max_val))
        else:
            detected_defects.append((template_label, temp_max_val))
        
    return (detected_parts, detected_defects)


def detectBestImage(images_corrected_angle, matching_thresh):
    best_image = None
    detected_parts_count = 0
    detected_defect_count = 0
    detected_parts = []
    detected_defects = []

    for image in images_corrected_angle:
        (temp_detected_parts, temp_detected_defects) = detectDefects(image, matching_thresh)

        size = len(temp_detected_parts)
        if(size > detected_parts_count):
            best_image = image
            detected_parts_count = size
            detected_parts = temp_detected_parts
            detected_defects = temp_detected_defects

    return (best_image, detected_parts_count, detected_parts, detected_defects)

def pipeline(img = None, defects = None):
    
    debug_level = DebugLevel.PRODUCTION

    print("PIPELINE STARTED")
    imgbackground = cv2.imread('./img/Other/image_100.jpg')
    SE_size = 5
    clearBorderRadius= 10
    matching_thresh = 0.3

    print("PREPROCESS IMAGE")
    img_no_background = subtractBackground(debug_level, img, imgbackground)
    cv2.imshow("img_no_background",img_no_background)
    img_no_background_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_no_background_gray",img_no_background_gray)
    img_no_background_binary = convertToBinary(debug_level, img_no_background)
    cv2.imshow("img_no_background_binary",img_no_background_binary)
    img_no_background_binary_floodFill = floodFill(img_no_background_binary)
    cv2.imshow("img_no_background_binary_floodFill",img_no_background_binary_floodFill)
    img_no_background_binary_floodFill_opened = opening(debug_level, img_no_background_binary_floodFill, SE_size=SE_size)
    cv2.imshow("img_no_background_binary_floodFill_opened",img_no_background_binary_floodFill_opened)
    img_no_background_binary_floodFill_opened_no_border = clearBorder(debug_level, img_no_background_binary_floodFill_opened, clearBorderRadius)
    cv2.imshow("img_no_background_binary_floodFill_opened_no_border",img_no_background_binary_floodFill_opened_no_border)
    img_no_background_binary_floodFill_opened_no_border_closed = closing(debug_level= DebugLevel.PRODUCTION, image=img_no_background_binary_floodFill_opened_no_border, SE_size=SE_size)
    cv2.imshow("img_no_background_binary_floodFill_opened_no_border_closed", img_no_background_binary_floodFill_opened_no_border_closed)
    

    count_non_zero = cv2.countNonZero(img_no_background_binary_floodFill_opened_no_border_closed)
    if(count_non_zero <100):
        print("IMAGE EMPTY")

        return
    

    print("IMAGE OKAY")
    print("LOCATE RECTANGLE")
    img_props = imutils.regionprops(img_no_background_binary_floodFill_opened_no_border_closed)
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
    
    # normal and cropped images are returned
    images_corrected_angle, cropped_images_corrected_angle = rotate(image = img_no_background_gray, cx = cx, cy = cy, angles = angles, rect = rect)
    
    # use cropped for further processing
    # CROPPED currently results in worse results
    # comment out next line if thats a problem
    images_corrected_angle = cropped_images_corrected_angle
    

    # show corrected angles
    cv2.imshow("images_corrected_angle[0]", images_corrected_angle[0])
    cv2.imshow("images_corrected_angle[1]", images_corrected_angle[1])
    cv2.imshow("images_corrected_angle[2]", images_corrected_angle[2])
    cv2.imshow("images_corrected_angle[3]", images_corrected_angle[3])

    print("TEMPLATE MATCHING")

    (best_image, detected_parts_count, detected_parts, detected_defects) = detectBestImage(images_corrected_angle, matching_thresh)
    print("BEST IMAGE PARTS DETECTED: " + str(detected_parts_count))
    
    if(best_image is not None):
        cv2.imshow("BEST IMAGE", best_image)
    
    if(detected_parts_count >= 2):
        print("=========== INDY  FOUND ===========")
    else:
        print("===========  NOT INDY  ===========")

    max_defect_label = None
    max_defect_val = 0

    for defect_label, defect_val in detected_defects:
        if(defect_val > max_defect_val):
            max_defect_label = defect_label
            max_defect_val = defect_val

    print("END")
    return img, max_defect_label

if __name__ == "__main__":
    imageDir = "./img/All/"

    for imagePath in glob.glob(imageDir + "*.jpg"):
        print("IMAGE: " +str(imagePath) + "\n")
        img = cv2.imread(imagePath)
        pipeline(img = img)


# "C:\development\Hagenberg\DIP\DIP_Projekt\img\All\image_100.jpg"
    
    # imageDir = "./img/All/image_100.jpg"
    # img = cv2.imread(imageDir)
    # pipeline(DebugLevel.PRODUCTION, img = img)