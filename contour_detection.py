import cv2
import numpy as np
import os

def box_extraction(file_path, cropped_dir_path):
    img = cv2.imread(file_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    cv2.imwrite("Image_bin.jpg", img_bin)
    
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
    
    # A vertical kernel of (1 x kernel_length), which will detect all the vertical lines in the image
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    
    # A horizontal kernel of (kernel_length x 1), which will help to detect all the horizontal lines in the image
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # A kernel of (3 x 3) ones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Morphological operation to detect vertical lines in an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)
    cv2.imwrite("vertical_lines.jpg", vertical_lines_img)
    
    # Morphological operation to detect horizontal lines in an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)
    
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image
    alpha = 0.5
    beta = 1.0 - alpha
    
    # This function helps to add two images with specific weight parameters to get a third image as a summation of the two images
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # For debugging
    # Enable this line to see vertical and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg", img_final_bin)
    
    # Find contours in the image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort all the contours from top to bottom
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
    
    idx = 0
    os.makedirs(cropped_dir_path, exist_ok=True)
    digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 20 and h >= 20 and 0.75*w <= h <= 1.25*w:
            idx += 1
            new_img = cv2.resize(img[y:y+h, x:x+w], (28, 28))
            # (thresh, new_img) = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            new_img_path = cropped_dir_path + '\\' + str(idx) + '.png'
            if not cv2.imwrite(new_img_path, new_img):
                raise Exception("Could not write image")
            digits.append(new_img/255)
            print(new_img_path, 'saved!')
                
    return digits

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

digits = box_extraction("number_0.jpg", "cropped")
# print(digits)
