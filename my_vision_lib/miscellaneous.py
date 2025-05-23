# My vision library
# By Tomasz Gołaszewski
# 2025.04

import numpy as np
import cv2
import time 

def get_objects_by_color(image_original: cv2.typing.MatLike, area_threshold: int = 5000) -> tuple[cv2.typing.MatLike, list]:
    """
    Method detects objects from passed image and returns masked image and list of found objects coordinates.

    Args:
        image_original (MatLike): Original image.
        area_threshold (int): The limit size of the area that defines the found object.

    Returns:
        MatLike: masked image with drawn objects.
        list: list of coordinates of found objects.
    """

    found_objects_list = []

    # convert the image_original in BGR (RGB color space) to HSV (hue-saturation-value color space) 
    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)

    # set range for red color and define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    mask_original = cv2.inRange(image_hsv, red_lower, red_upper)
    # cv2.imshow('red_mask', mask_original) # show window for testing purposes # TODO: to remove

    # morphological transform, dilation for each color and bitwise_and operator
    # between image_original and mask determines to detect only that particular color

    # prepares transformation template, each pixel will be enlarged 5 times
    kernel = np.ones((5, 5), "uint8")

    # pixel enlargement
    mask_enlarged = cv2.dilate(mask_original, kernel)
    # cv2.imshow('red_mask_2', mask_enlarged) # show window for testing purposes # TODO: to remove

    # masks original image with prepared mask with enlarged pixels
    image_masked = cv2.bitwise_and(image_original, image_original, 
                            mask = mask_enlarged)
    # cv2.imshow('res_red', image_masked) # show window for testing purposes # TODO: to remove

    # creating contour to track color
    contours, hierarchy = cv2.findContours(mask_enlarged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > area_threshold):
            x, y, w, h = cv2.boundingRect(contour)
            center_coordinates = (x+w//2, y+h//2)
            found_objects_list.append(center_coordinates)
            # draw rectangle
            image_masked = cv2.rectangle(image_masked, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # draw circle
            image_masked = cv2.circle(image_masked, center_coordinates, 10, (0, 0, 255), 2)

    return image_masked, found_objects_list


def draw_points_from_list(image: cv2.typing.MatLike, 
                          points: list, 
                          color: tuple = (0, 0, 255), 
                          radius: int = 10, 
                          border: int = 2):
    """
    Draws circles on an image at specified points.

    Args:
        image (MatLike): The input image on which circles will be drawn.
        points (list): A list of points where each point is represented as [x, y].
        color (tuple, optional): The color of the circles in BGR format. Default is (0, 0, 255) (red).
        radius (int, optional): The radius of the circles. Default is 10.
        border (int, optional): The thickness of the circle border. Default is 2.

    Returns:
        MatLike: The image with the circles drawn on it.
    """
    for p in points:
        image = cv2.circle(image, [int(p[0]), int(p[1])], radius, color, border)
    return image