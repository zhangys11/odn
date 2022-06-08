# %load batch_object_detection.py
# %load batch_object_detection.py
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six

from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import importlib
importlib.reload(vis_util) # reflect changes in the source file immediately

from tqdm import tqdm

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

from math import sqrt
from math import pow

def calculate_fundus_zones(classes, scores, boxes, threshold):
    idxOD = -1
    idxOS = -1
    idxMacula = -1
    
    idx = 0
                    
    for c in classes:                          
        if (c == 1 and scores[idx] > threshold):
            idxOD = idx
        if (c == 2 and scores[idx] > threshold):
            idxOS = idx
        if (c == 3 and scores[idx] > threshold):
            idxMacula = idx
        idx = idx + 1
    
    cx = -1
    cy = -1
    if (idxOD >= 0):
        cx = (boxes[idxOD][3] + boxes[idxOD][1])/2.0
        cy = (boxes[idxOD][2] + boxes[idxOD][0])/2.0
            
    if (idxOS >= 0):
        cx = (boxes[idxOS][3] + boxes[idxOS][1])/2.0
        cy = (boxes[idxOS][2] + boxes[idxOS][0])/2.0
        
    if (cx == -1):
        return []    
    
    cx_m = -1
    cy_m = -1

    if (idxMacula >= 0):
        cx_m = (boxes[idxMacula][3] + boxes[idxMacula][1])/2.0
        cy_m = (boxes[idxMacula][2] + boxes[idxMacula][0])/2.0
        
    radius = 0.5
    if (cx_m != -1):
        radius = 2*( sqrt((cx-cx_m)**2 + (cy-cy_m)**2) )            
   
    # (xmin, xmax, ymin, ymax) 
    zone1 = [cx-radius, cx+radius, cy-radius*4/3, cy+radius*4/3]
    zone2 = [cx-2*radius, cx+2*radius, cy-2*radius*4/3, cy+2*radius*4/3]
    
    return [zone1, zone2]     

def draw_fundus_zones_on_image_array(image, zones, text ='', use_normalized_coordinates=True):
    
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    
    colors = ["orange", "yellow"]
    idx = 0
    
    for zone in zones:
        # (ymin, xmin, ymax, xmax)
        xmin = zone[0]
        ymin = zone[2]
        xmax = zone[1]
        ymax = zone[3]
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)      
     
        draw.ellipse((int(left), int(top), int(right),int(bottom)), fill = None, outline =colors[idx])   # (255,255,255)
        
        idx+=1
        
    try:
        font = ImageFont.truetype('arial.ttf', 12)
    except IOError:
        font = ImageFont.load_default()

    draw.text(
        (20,20),
        text,
        fill=(255,255,255),
        font=font)    
    
    np.copyto(image, np.array(image_pil))


def batch_object_detection(detection_graph, category_index, FILES, 
                           target_folder, log_file, 
                           display = False, savefile = True, 
                           IMAGE_SIZE = (24, 18), threshold = 0.2, 
                           new_img_width = None,
                          fontsize = 24):    
    if savefile:
        os.makedirs(target_folder, exist_ok=True) # create the target folder if not exist
    
    if (log_file is not None and log_file != ''):
        if os.path.exists(log_file):
            os.remove(log_file)
        try:
            open(log_file, 'w')
        except OSError:
            print('Warning: log file path ', log_file, ' is invalid')
            log_file = None
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            with tqdm(total=len(FILES)) as pbar:
                for image_path in FILES:
                    image = Image.open(image_path)
                    if new_img_width > 0:
                        image.thumbnail((new_img_width,new_img_width)) # Image.ANTIALIAS
                    # the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                        
                    # get image dims
                    (im_width, im_height) = image.size
                        
                    # special treatment: only keep one opticdisk: OD or OS
                    idxOD = -1
                    idxOS = -1
                    idxMacula = -1
                    probMacula = 0
                    idx = 0
                    
                    # log object detection info
                    info = os.path.basename(image_path) + ' '                    
                    
                    for c in classes[0]:                          
                        
                        info = info + str(int(c)) + ' ' + str(round(scores[0][idx],3)) + ' '
                        
                        if (c == 1):
                            idxOD = idx
                        if (c == 2):
                            idxOS = idx
                        if (c == 3):
                            if idxMacula == -1 or probMacula < scores[0][idx]:
                                if idxMacula > 0:
                                    scores[0][idxMacula] = 0.0 
                                idxMacula = idx
                                probMacula = scores[0][idx]
                                
                        idx = idx + 1
                    
                    if display: 
                        print('Top Objects:\n', 'boxes = ', boxes[0], '\nscores = ', scores[0], '\nclasses = ', classes[0])
                    
                   
                    ###### RULE1 #######
                    # Discard oversized macula candidates; Judge the distance between macula and OpticDisk                       
                    # On a 512-pixel-high image, macula radius is 55, optic disk radius is 35
                    # The boxes object is a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The coordinates are in normalized format between [0, 1].
                    m_w = abs(boxes[0][idxMacula][3] - boxes[0][idxMacula][1])
                    m_h = abs(boxes[0][idxMacula][2] - boxes[0][idxMacula][0])
                    m_cx = abs(boxes[0][idxMacula][3] + boxes[0][idxMacula][1])/2.0
                    m_cy = abs(boxes[0][idxMacula][2] + boxes[0][idxMacula][0])/2.0
                    
                    # print(m_w, m_h, m_cx, m_cy)
                    
                    if m_w > 0.3 or m_h > 0.3:
                        scores[0][idxMacula] = 0.0
                    
                    
                    ##### RULE2: Optic Disk OD/OS Judgment ######
                    od_cx = abs(boxes[0][idxOD][3] + boxes[0][idxOD][1])/2.0
                    od_cy = abs(boxes[0][idxOD][2] + boxes[0][idxOD][0])/2.0
                            
                    os_cx = abs(boxes[0][idxOS][3] + boxes[0][idxOS][1])/2.0
                    os_cy = abs(boxes[0][idxOS][2] + boxes[0][idxOS][0])/2.0
                            
                    # OpticDisk near the left rim is OS
                    if (scores[0][idxOD] > threshold and scores[0][idxOD]>scores[0][idxOS] and od_cx < 0.2): # 2.0*35*2/512
                        scores[0][idxOS] = max(scores[0][idxOS], scores[0][idxOD])
                        scores[0][idxOD] = 0
                        boxes[0][idxOS] = boxes[0][idxOD] # ?üD?bbox
                        #classes[0][idxOD] = 2 # set as OS
                            
                    # OpticDisk near the right rim is OD
                    if (scores[0][idxOS] > threshold  and scores[0][idxOD]<scores[0][idxOS] and os_cx > 0.8):
                        scores[0][idxOD] = max(scores[0][idxOS], scores[0][idxOD])
                        scores[0][idxOS] = 0
                        boxes[0][idxOD] = boxes[0][idxOS] # ?üD?bbox
                        #classes[0][idxOS] = 1 # set as OD
                        
                    
                    ##### RULE3: Judge relative positions of macula and OpticDisk
                    
                    # reload boxes info
                    od_cx = abs(boxes[0][idxOD][3] + boxes[0][idxOD][1])/2.0
                    od_cy = abs(boxes[0][idxOD][2] + boxes[0][idxOD][0])/2.0
                            
                    os_cx = abs(boxes[0][idxOS][3] + boxes[0][idxOS][1])/2.0
                    os_cy = abs(boxes[0][idxOS][2] + boxes[0][idxOS][0])/2.0
                    
                    if (scores[0][idxMacula] > threshold):
                        if (scores[0][idxOD] > threshold and scores[0][idxOD] > scores[0][idxOS] and m_cx > od_cx ):
                            scores[0][idxOS] = max(scores[0][idxOS], scores[0][idxOD])
                            scores[0][idxOD] = 0
                            boxes[0][idxOS] = boxes[0][idxOD]
                        if (scores[0][idxOS] > threshold and scores[0][idxOD] < scores[0][idxOS] and m_cx < os_cx):
                            scores[0][idxOD] = max(scores[0][idxOS], scores[0][idxOD])
                            scores[0][idxOS] = 0
                            boxes[0][idxOD] = boxes[0][idxOS] # update bbox
                    
                    
                    ##### RULE4: Keep the bigger probality of OpticDisk
                    
                    if (scores[0][idxOD] > scores[0][idxOS]):                        
                        scores[0][idxOS] = 0.0                        
                    if (scores[0][idxOD] < scores[0][idxOS]):
                        scores[0][idxOD] = 0.0                                    
                        

                    info += ' ; ' 
                    idx = 0
                    label = ''
                    for c in classes[0]:                        
                        info = info + str(round(scores[0][idx],3)) + ' '                        
                        if (scores[0][idx] > threshold):
                            label += category_index[c]['name'] + ' ' + str(round(scores[0][idx],3)) + '  '
                        idx += 1
                                       
                    zones = calculate_fundus_zones(classes[0], scores[0], boxes[0], threshold)
                    draw_fundus_zones_on_image_array(image_np, zones = zones, text=label)
                    
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1, # 3   
                        max_boxes_to_draw=2, # 3
                        min_score_thresh=threshold,
                        fontsize = fontsize)
                    fig = plt.figure(figsize=IMAGE_SIZE)
                    if (display):
                        plt.imshow(image_np)
                    if(savefile):
                        plt.imsave(os.path.join(target_folder, os.path.basename(image_path)), image_np)
                        # plt.annotate(info, (0, 0), color='b', weight='bold', fontsize=12, ha='left', va='top')
                        # print(info)
                        plt.close(fig)
                        
                    if (log_file is not None and log_file != ''): # the validity of log_file is already checked in the beginning
                        with open(log_file, "a") as myfile:
                            myfile.write(info + '\n')                        
                    pbar.update(1)                        