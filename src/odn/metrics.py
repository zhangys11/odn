import numpy as np
import math

def bbox_IoU(boxA, boxB):
    
    # the two bboxes don't overlap
    if (boxA[0] >= boxB[2] or boxA[2] <= boxB[0]
       or boxA[1] >= boxB[3] or boxA[3] <= boxB[1]):
        return 0    
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if(boxAArea == 0 or boxBArea ==0):
        return 0

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bbox_RCE(boxA, boxB, width, height):
    '''
	RCE (Relative Center Error): 
	Calculate the Euclid distance divided by the image diagonal length.
	'''

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if(boxAArea == 0 or boxBArea ==0):
        return 1 # 1 is the biggest error value
    
    # determine the (x, y)-coordinates of the intersection rectangle
    cxA = (boxA[2] + boxA[0]) / 2 
    cyA = (boxA[3] + boxA[1]) / 2
    cxB = (boxB[2] + boxB[0]) / 2 
    cyB = (boxB[3] + boxB[1]) / 2
    
    diag = math.sqrt((width)**2 + (height)**2)
    
    # compute the rectangle area formed by the prediction and ground-truth ROI centers.
    deltaArea = math.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)

    return deltaArea / diag