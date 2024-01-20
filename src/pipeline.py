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

def start_pipeline(debug_level = DebugLevel.DEBUG, img = None):

    template, defects = init.initdata()

    # cv2.imshow("name",template["img_mask_no_hat"])
    # cv2.waitKey()

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
        img = normal_imgs[12]
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

    # apply closing operation using a SE
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # borderType is constant by default
    closing = cv2.morphologyEx(im_opened, cv2.MORPH_CLOSE, se, iterations=1)
    showImage(closing, debug_level=debug_level,name="Img closing")


    # extract rectangle properties
    img_props = imutils.regionprops(closing)
    contours, area_vec, [cx, cy], rect, ell_rot = img_props
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # width, height = rect[1][0], rect[1][1]
    # angle = rect[2]
    
    # if width > height:
    #     angle = 90 + angle  # Adjust angle

    # Draw the rectangle on the original image/copy
    image_with_rect = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)
    # cv2.ellipse(image_with_rect,rect, ell_rot, (128,128,0),2)
    showImage(image_with_rect, debug_level=debug_level,name="Rectangle")

    # Calculate the rotation matrix
    
    M = cv2.getRotationMatrix2D((cx, cy), rect[2], 1.0)
    M180 = cv2.getRotationMatrix2D((cx, cy), rect[2] + 180, 1.0)

    # M = cv2.getRotationMatrix2D((cx, cy), 180 + ell_rot, 1.0)
    # M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, image.shape[1::-1])
    rotated_image_bw = cv2.warpAffine(closing, M, image.shape[1::-1])
    rotated_image_bw_180 = cv2.warpAffine(closing, M180, image.shape[1::-1])

    # Get the size of the rotated rectangle
    size = (int(rect[1][0]), int(rect[1][1]))

    # Extract the straightened ROI
    x, y = np.int0((cx, cy))
    x -= size[0] // 2
    y -= size[1] // 2
    straightened_roi = rotated_image[y:y+size[1], x:x+size[0]]
    straightened_roi_bw = rotated_image_bw[y:y+size[1], x:x+size[0]]
    straightened_roi_bw_180 = rotated_image_bw_180[y:y+size[1], x:x+size[0]]

    showImage(image, debug_level=DebugLevel.DEBUG,name="Normal")
    # showImage(rotated_image, debug_level=DebugLevel.DEBUG,name="Rotated")
    showImage(straightened_roi, debug_level=debug_level,name="sub image Rotated")
    showImage(straightened_roi_bw, debug_level=debug_level,name="sub image Rotated")
    showImage(straightened_roi_bw_180, debug_level=debug_level,name="sub image Rotated")

    straightened_roi_bw, template["img_mask"] = pad_to_center(straightened_roi_bw, template["img_mask"])
    straightened_roi_bw_180, template["img_mask"] = pad_to_center(straightened_roi_bw_180, template["img_mask"])
    
    # test = straightened_roi_bw - template["img_mask"]
    bitwise_and = cv2.bitwise_and(straightened_roi_bw, template["img_mask"])
    bitwise_and_180 = cv2.bitwise_and(straightened_roi_bw_180, template["img_mask"])

    showImage(straightened_roi_bw, debug_level=debug_level,name="subtracted template")
    showImage(template["img_mask"], debug_level=debug_level,name="subtracted template")
    showImage(bitwise_and, debug_level=debug_level,name="subtracted template")
    showImage(bitwise_and_180, debug_level=debug_level,name="180 subtracted template")

    print("rotated correctly:", cv2.countNonZero(bitwise_and) > cv2.countNonZero(bitwise_and_180))

    if(cv2.countNonZero(bitwise_and) > cv2.countNonZero(bitwise_and_180)):
        rotated_image = cv2.warpAffine(image, M, image.shape[1::-1])
    else:
        rotated_image = cv2.warpAffine(image, M180, image.shape[1::-1])

    showImage(rotated_image, debug_level=DebugLevel.DEBUG,name="180 subtracted template")


if __name__ == "__main__":
    start_pipeline(DebugLevel.PRODUCTION)