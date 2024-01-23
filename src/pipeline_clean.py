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
    # 230 works good till no body print
    (thresh, imgBW) = cv2.threshold(im_gray, 220, 255, cv2.THRESH_BINARY_INV)
    
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
            if width < 144:
                width = 144
            if height < 220:
                height = 220
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

    # print("image shape " + str(image.shape))
    # print("template_image shape " + str(template_image.shape))
    # print("template_mask shape " + str(template_mask.shape))

    # apply zero-mean cross correlation
    res = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED, mask=template_mask)
    # search for highest match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # print("Zero-Mean Cross Correlation Coefficients ranges from {} to {}".format(min_val, max_val))
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

def predict_label(image, defects, template_gray):
    # defect needs smallest number
    min_val = 1
    defect_class_label = -1
    for class_label, defect_type in enumerate(defects):
        template_mask = defects[defect_type]["mask"]
        temp_max_val = matchTemplateWithVariation(image, template_gray, template_mask, 5)
        if(temp_max_val <= min_val):
            min_val = temp_max_val
            defect_class_label = class_label
    
    return (defect_class_label, min_val)

def is_missing(image, template_gray, template_mask, threshold):
    temp_max_val = matchTemplateWithVariation(image, template_gray, template_mask, 5)

    return (temp_max_val < threshold, abs(temp_max_val - threshold))


def pipeline(img = None, defects = None):
    
    # todo: do not override defects if not None
    template, _ = init.initdata()

    debug_level = DebugLevel.PRODUCTION

    print("PIPELINE STARTED")
    imgbackground = cv2.imread('./img/Other/image_100.jpg')
    SE_size = 20
    clearBorderRadius= 5
    matching_thresh = 0.3

    print("PREPROCESS IMAGE")
    img_no_background = subtractBackground(debug_level, img, imgbackground)
    cv2.imshow("img_no_background",img_no_background)
    img_no_background_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_no_background_gray",img_no_background_gray)
    img_no_background_binary = convertToBinary(debug_level, img_no_background)
    img_no_background_binary_floodFill = floodFill(img_no_background_binary)
    img_no_background_binary_floodFill_no_border = clearBorder(debug_level, img_no_background_binary_floodFill, clearBorderRadius)
    

    count_non_zero = cv2.countNonZero(img_no_background_binary_floodFill_no_border)
    print("count_non_zero",count_non_zero)
    if(count_non_zero <1000):
        print("IMAGE EMPTY")

        return img , 0
    

    print("IMAGE OKAY")
    print("LOCATE RECTANGLE")
    img_props = imutils.regionprops(img_no_background_binary_floodFill_no_border)
    contours, area_vec, [cx, cy], rect, ellipse = img_props
    
    print("CORRECT RECTANGLE")
    rect = correct90DegreeRectangle(rect)
    print(rect)
    img_no_background_gray_rectangle = drawRectanle(image=img_no_background_gray, rect=rect)
    img_no_background_gray_ellipse = draw_ellipse(img_no_background_gray_rectangle, ellipse)
    cv2.imshow("img_no_background_gray_ellipse", img_no_background_gray_ellipse)


    print("ROTATE IMAGE")
    # also correct if rectangle is upside down
    angles = [ellipse[2], ellipse[2] + 180]
    
    # normal and cropped images are returned
    images_corrected_angle, cropped_images_corrected_angle = rotate(image = img_no_background_gray, cx = cx, cy = cy, angles = angles, rect = rect)
    
    # use cropped for further processing
    # CROPPED currently results in worse results
    # comment out next line if thats a problem
    images_corrected_angle = cropped_images_corrected_angle
    
    if(images_corrected_angle[0].shape[0] <= 0):
        print("RIP IMAGE")

        return (img, -1)

    print("GET CORRECT ROTATION")
    
    images_corrected_angle_max_val = 0
    better_image = images_corrected_angle[0]
    
    template_gray = cv2.cvtColor(template["img"], cv2.COLOR_BGR2GRAY )
    
    full_body_mask = template["mask"]
    hand_mask = defects["hand"]["mask"]
    arm_mask = defects["arm"]["mask"]
    leg_mask = defects["leg"]["mask"]
    hat_mask = defects["hat"]["mask"]
    face_print_mask = defects["face print"]["mask"]
    body_print_mask = defects["body print"]["mask"]
    head_mask = defects["head"]["mask"]
    mask_no_head = template["mask_no_head"]
    mask_no_neck = cv2.bitwise_and(mask_no_head, cv2.bitwise_not(head_mask))

    for image in images_corrected_angle:
        res = cv2.matchTemplate(image, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask_no_head)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print("Zero-Mean Cross Correlation Coefficients ranges from {} to {}".format(min_val, max_val))
        
        # if(max_val <= 0.3):
        res = cv2.matchTemplate(image, template_gray, cv2.TM_CCOEFF_NORMED, mask=arm_mask)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res)
        print("Zero-Mean Cross Correlation Coefficients ranges from {} to {}".format(min_val, max_val))
        
        max_val += max_val2

        print("max_val" + str(max_val))

        if(max_val != float('inf') and max_val > images_corrected_angle_max_val):
            images_corrected_angle_max_val = max_val
            better_image = image

    # Rotation should be fixed here (at least for all Normal indys)
    cv2.imwrite("better_image.jpg", better_image)
    better_image_for_print = cv2.resize(better_image, (better_image.shape[1] * 2, better_image.shape[0] * 2), interpolation=cv2.INTER_AREA)
    cv2.imshow("BEST ROTATION FOUND IMAGE", better_image_for_print)

    print("TEMPLATE MATCHING")
    print("Classify Image")
    
    defect = 0
    (is_hand_missing, is_hand_missing_distance) = is_missing(image, template_gray, hand_mask, 0.215)
    (is_arm_missing, is_arm_missing_distance) = is_missing(image, template_gray, arm_mask, 0.450)
    (is_leg_missing, is_leg_missing_distance) = is_missing(image, template_gray, leg_mask, 0.370)
    (is_hat_missing, is_hat_missing_distance) = is_missing(image, template_gray, hat_mask, 0.380)
    (is_face_print_missing, is_face_print_missing_distance) = is_missing(image, template_gray, face_print_mask, 0.5)
    (is_body_print_missing, is_body_print_missing_distance) = is_missing(image, template_gray, body_print_mask, 0.5)
    (is_head_missing, is_head_missing_distance) = is_missing(image, template_gray, head_mask, 0.38)
    
    max_distance = max(is_hand_missing_distance, is_arm_missing_distance, is_leg_missing_distance, is_hat_missing_distance, is_body_print_missing_distance,is_head_missing_distance)

    if(is_hand_missing and is_hand_missing_distance == max_distance):
        defect = 0
    if(is_arm_missing and is_arm_missing_distance == max_distance):
        defect = 1
    if(is_leg_missing and is_leg_missing_distance == max_distance):
        defect = 2
    if(is_hat_missing and is_hat_missing_distance == max_distance):
        defect = 3
    if(is_body_print_missing and is_body_print_missing_distance == max_distance):
        defect = 5
    if(is_head_missing and is_head_missing_distance == max_distance):
        defect = 6
    print("")

    return img, defect

if __name__ == "__main__":
    # 0 hand
    # 1 arm
    # 2 leg
    template, defects = init.initdata()

    # imageDir = "./img/0-Normal/"
    # imageDir = "./img/1-NoHat/"
    # imageDir = "./img/2-NoFace/"
    # imageDir = "./img/3-NoLeg/"
    # imageDir = "./img/4-NoBodyPrint/"
    # imageDir = "./img/5-NoHand/"
    imageDir = "./img/6-NoHead/"
    # imageDir = "./img/7-NoArm/"
    # imageDir = "./img/All/"

    for imagePath in glob.glob(imageDir + "*.jpg"):
        print("IMAGE: " +str(imagePath) + "\n")
        img = cv2.imread(imagePath)
        pipeline(img = img, defects=defects)
