from http.client import UnimplementedFileMode
import numpy as np
import os
import os.path
import pathlib
import io
import time
import pickle
import matplotlib
import shutil
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from math import sqrt
from tqdm import tqdm
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import pandas as pd
import random
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import plotly.graph_objects as go
import plotly.express as px

import tensorflow as tf
from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from odn.tf_ssd.object_detection.utils import label_map_util
from odn.tf_ssd.object_detection.utils import visualization_utils as vis_util
import importlib
importlib.reload(vis_util) # reflect changes in the source file immediately

class demographics():
    '''
    Provides static ploting functions for subject demographics, Such As Gender, Birth Weight, Gestational Age, etc.
    '''

    def piechart_binary(values = [50, 50], labels= ['Group 1', 'Group 2'],
        colors= ['gold', 'mediumturquoise'] ):
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', textfont_size=20,
                                    marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                                    # insidetextorientation='radial' # this will cause text rotate
                                    )])
        fig.show()

    def piechart_preterm(values):
        demographics.piechart_binary(values = values, labels= ['Preterms', 'Non-preterms'])

    def piechart_gender(values = [50, 50]):        
        demographics.piechart_binary(values = values, labels= ['male', 'female'])

    def piechart_rop(values = [50, 50]):        
        demographics.piechart_binary(values = values, labels= ['ROP', 'non-ROP'])

    def hist_gw(ga_week):
        '''
        Plot gestational age histgram in weeks
        '''
        # create the bins
        hist, bins = np.histogram(ga_week, bins = 40, range=(22,42), density = True)
        # hist = np.append(hist, 0)
        # bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x = bins[:-1], y=hist/hist.sum() * 100, labels={'x':'Gestational Age (Weeks)', 'y':'Percentage (%)'})
        # fig = px.histogram(data, x="total_bill", histnorm='probability density')
        fig.update_layout(plot_bgcolor = "white")
        fig.update_traces(marker_color='mediumturquoise', marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.8)
        fig.show()

    def hist_bw(bw_kg):
        '''
        Plot birth weight histgram in kg
        '''
        # create the bins
        hist, bins = np.histogram(bw_kg, bins = 21, range=(1,2.6), density = True)
        # hist = np.insert(hist, -1, 0)
        # bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x = bins[:-1], y=hist/hist.sum() * 100, labels={'x':'Birth Weight(kg)', 'y':'Percentage (%)'})
        # fig = px.histogram(data, x="total_bill", histnorm='probability density')
        fig.update_layout(plot_bgcolor = "white")
        fig.update_traces(marker_color='mediumturquoise', marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.6)
        fig.show()


class annotation():

    def get_bbox_of_circle(cx, cy, h, c):

        '''
        Parameters
        ----------
        cx, cy : center of circle
        h : image height
        c : "opticDisk" or "Macula"

        Remarks
        -------
        The fundus images are fixed in 4:3 ratio.
        On a 512-pixel-high image, macula radius is 55, optic disk radius is 35
        '''
        if (c =='OpticDisk'):
            r = 35./512*h
        elif c == 'Macula':
            r = 55./512*h
        xmin = round(cx - r)
        xmax = round(cx + r)
        ymin = round(cy -r)
        ymax = round(cy + r)
        return xmin,ymin,xmax,ymax  

    def show_anno(filepath, df, savefolder = None, drawzones = False, showimg = True):    
        '''
        Display an image with the annotations. 

        Parameters
        ----------
        filepath : image file path
        df : pandas dataframe of the annotation csv file in VIA format. 

                This is the VIA annotation format
                    filename: file name of the image
                    class: denotes the class label of the ROI (region of interest)
                    cx: image width
                    cy: image height
                    xmin: x-coordinate of the bottom left part of the ROI
                    xmax: x-coordinate of the top right part of the ROI
                    ymin: y-coordinate of the bottom left part of the ROI
                    ymax: y-coordinate of the top right part of the ROI  
                    laterality: L001 = OD, L002 = OS

        saveFolder : A target file path to save the image with annotation. 
            If None, will not save.
        showing : whether display the annotated image inline.

        '''

        # restore font size
        plt.rcParams.update({'font.size': 10})    
        
        filename = os.path.basename(filepath) # file name
        
        fig = plt.figure()
        image = plt.imread(filepath)
        plt.imshow(image)
        ax = fig.gca()
        ax.set_axis_off()

        idxs = []
        base_fn = os.path.basename(filename).lower()
        for idx, fn in enumerate(df.filename):
            base_fn2 = os.path.basename(fn).lower()
            if (base_fn == base_fn2):
                idxs.append(idx)

        # print(df.iloc[idxs])

        cx = -1
        cy = -1
        cx_m = -1
        cy_m = -1

        for _, row in df.iloc[idxs].iterrows():
            # print(row)
            xmin = row.xmin
            xmax = row.xmax
            ymin = row.ymin
            ymax = row.ymax

            width = xmax - xmin
            height = ymax - ymin

            edgecolor = 'white'
            prob = ''
            if ('prob' in df.columns):
                prob = ' ' + str(round(row['prob'], 3))
            
            # assign different color to different classes of objects
            if row['class'] == 'OpticDisk':
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                edgecolor = 'white'
            elif row['class'] == 'Macula':
                cx_m = (xmin + xmax) / 2.0
                cy_m = (ymin + ymax) / 2.0
                edgecolor = 'lavender' #'azure' 
            
            radius = 0.5
            if (cx_m != -1):
                radius = 2*( sqrt((cx-cx_m)**2 + (cy-cy_m)**2) )            
        
            # (xmin, xmax, ymin, ymax) 
            zone1 = [cx-radius, cx+radius, cy-radius*4/3, cy+radius*4/3]
            zone2p = [cx-1.3*radius, cx+1.3*radius, cy-1.3*radius*4/3, cy+1.3*radius*4/3]
            zone2 = [cx-2*radius, cx+2*radius, cy-2*radius*4/3, cy+2*radius*4/3]

            # calculate the position of anno labels
            xy = (max(1, xmin),ymin-5)
            if (ymin < 5):
                xy = (max(1, xmin),ymax)
            ax.annotate(row['class'] + prob, xy=xy, color = edgecolor)

            # add bounding boxes to the image
            rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')

            ax.add_patch(rect)

        if drawzones:
            # buf = io.BytesIO()
            # fig.savefig(buf)
            # buf.seek(0)
            # image = Image.open(buf)
            # # image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            # annotation.draw_fundus_zones_on_image_array(image, zones = [zone1, zone2p, zone2])
            # plt.imshow(image)
            
            for zone, color in zip([zone1, zone2p, zone2], ["orange", "gold", "yellow"]):
                # (ymin, xmin, ymax, xmax)
                xmin = zone[0]
                ymin = zone[2]
                xmax = zone[1]
                ymax = zone[3]               
            
                ellipse = patches.Ellipse((cx, cy), xmax-xmin , ymax-ymin, edgecolor = color, facecolor = 'none')   # (255,255,255)
                ax.add_patch(ellipse)

        if savefolder:
            if (os.path.isdir(savefolder) == False):
                os.makedirs(savefolder)
            plt.savefig(savefolder + '/' + filename)
        
        if showimg == False:
            plt.close(fig)


    def rule_filter_rois(input_file = '../src/odn/candidate_rois.txt', 
    output_file = '../src/odn/rois.txt',
    verbose = True):

        rois_raw = pd.read_csv(input_file)
    
        if verbose:
            print('\n-------- \nInput file: ' + input_file)
            print(rois_raw.head())

        rois = rois_raw.copy()
        pd.options.mode.chained_assignment = None  # disable the "A value is trying to be set on a copy of a slice from a DataFrame" warning

        Dod = 0.1 # diameter of optic disc
        Dma = 0.16 # diameter of macula
        Dom = 0.32 # distance between optic disc and macula

        uniquefiles = set(rois['filename'].values)
        all_to_be_removed = []

        for index, fn in enumerate(uniquefiles):
            # print(fn)
            # filepath = DIR_IMAGES + '/' + fn
            rows = rois.loc[rois.filename == fn]
            ods = rows.loc[rows['class'] == 'OpticDisk']
            mas = rows.loc[rows['class'] == 'Macula']    
            to_be_removed = []
            
            if (len(ods) > 0):
                # use the max-probability optic disc
                od = ods.loc[ods['prob'].idxmax()]
                # rois.drop(ods.index[ods.index != ods['prob'].idxmax()]) # remove non-max prob bboxes
                to_be_removed = np.append(to_be_removed, ods[ods.index != ods['prob'].idxmax()].seq.values.flatten()) 
                d_od = (od['xmax'] - od['xmin']) / od['width']
                
                assert (abs(d_od - Dod) <= Dod * 0.5)
                cx_od = (od['xmax'] + od['xmin']) / 2
                cy_od = (od['ymax'] + od['ymin']) / 2
                
                if len(mas) > 0:
                
                    ma_dict={}
                    idx_least_error = -1
                    for idx,ma in mas.iterrows():            
                        cx_ma = (ma['xmax'] + ma['xmin']) / 2
                        cy_ma = (ma['ymax'] + ma['ymin']) / 2

                        dom = abs((cx_ma - cx_od) / od['width'])
                        err_dom = abs(dom - Dom)
                        if idx_least_error == -1:                
                            idx_least_error = idx
                        else:
                            if (ma_dict[idx_least_error] > err_dom):
                                idx_least_error = idx

                        ma_dict[idx] = err_dom            
                    
                    # print(ma_dict)
                    to_be_removed = np.append(to_be_removed, mas[mas.index != idx_least_error].seq.values.flatten())
                    # rois.drop(mas.index[mas.index != idx_least_error]) # only keep the region with the least error     
                    
                    ma = mas.loc[idx_least_error]
                    cx_ma = (ma['xmax'] + ma['xmin']) / 2
                    cy_ma = (ma['ymax'] + ma['ymin']) / 2           
                    dom = abs((cx_ma - cx_od) / od['width'])
                    err_dom = abs(dom - Dom)
                    
                    
                    if ('laterality' in od):
                    
                        if ((od['laterality'] == 'L001' and cx_ma >= cx_od) #  For right eye (OD) image, macula should locate on the left side of optic disc
                            or (od['laterality'] == 'L002' and cx_ma <= cx_od)): # For left eye (OS) image, macula should locate on the right side of optic disc
                            to_be_removed = np.append(to_be_removed, ma.seq)
                        
                    
                    delta_y = abs(cy_ma - cy_od) / od['width']
                    
                    if (err_dom > 0.2 or delta_y > 0.2): # max allowed error 20%
                        to_be_removed = np.append(to_be_removed, ma.seq)
                        
                
            else:
                # no optic disc is detected. Keep the maximum-prob macula 
                if (len(mas) > 0):
                    # print(mas.index[mas.index != mas['prob'].idxmax()])
                    # rois.drop(mas.index[mas.index != mas['prob'].idxmax()]) # remove non-max prob bboxes
                    to_be_removed = np.append(to_be_removed, mas[mas.index != mas['prob'].idxmax()].seq.values.flatten())
            
            rows.drop(rows.loc[rows.seq.isin(to_be_removed)].index, inplace=True)
            all_to_be_removed = np.concatenate((all_to_be_removed, to_be_removed))
            
            #d_od = row['xmax'] - row['xmin']
            #show_anno(filepath, all, './fundus_image_dataset/ground_truth/', savefile = False, showimg = True)
            #show_anno(filepath, rows, './fundus_image_dataset/odn_10e/', savefile = True, showimg = False)

        if verbose:
            print('\n-------- \nIndices of removed annos: ', all_to_be_removed)

        rois.drop(rois.loc[rois.seq.isin(all_to_be_removed)].index, inplace=True)
        rois.to_csv(output_file)

        if verbose:
            print('\n-------- \nSaved to ', output_file)
            rois= pd.read_csv(output_file)
            print(rois.head())

        return rois


    def naive_filter_rois(input_file = '../src/odn/candidate_rois.txt', 
    output_file = '../src/odn/rois_naive.txt',
    verbose = True):

        rois_raw = pd.read_csv(input_file)
    
        if verbose:
            print('\n-------- \nInput file: ' + input_file)
            print(rois_raw.head())

        rois = rois_raw.copy()
        pd.options.mode.chained_assignment = None  # disable the "A value is trying to be set on a copy of a slice from a DataFrame" warning

        Dod = 0.1 # diameter of optic disc
        Dma = 0.16 # diameter of macula
        Dom = 0.32 # distance between optic disc and macula

        uniquefiles = set(rois['filename'].values)
        all_to_be_removed = []

        for index, fn in enumerate(uniquefiles):
            # print(fn)
    
            rows = rois.loc[rois.filename == fn]
            ods = rows.loc[rows['class'] == 'OpticDisk']
            mas = rows.loc[rows['class'] == 'Macula']    
            to_be_removed = []
            
            if (len(ods) > 0):
                to_be_removed = np.append(to_be_removed, ods[ods.index != ods['prob'].idxmax()].seq.values.flatten()) 

            if (len(mas) > 0):
                to_be_removed = np.append(to_be_removed, mas[mas.index != mas['prob'].idxmax()].seq.values.flatten())
            
            rows.drop(rows.loc[rows.seq.isin(to_be_removed)].index, inplace=True)
            all_to_be_removed = np.concatenate((all_to_be_removed, to_be_removed))

        if verbose:
            print('\n-------- \nIndices of removed annos: ', all_to_be_removed)

        rois.drop(rois.loc[rois.seq.isin(all_to_be_removed)].index, inplace=True)
        rois.to_csv(output_file)

        if verbose:
            print('\n-------- \nSaved to ', output_file)
            rois= pd.read_csv(output_file)
            print(rois.head())

        return rois

    #------------------ region tf_ssd --------------------#

    def calculate_fundus_zones(classes, scores, boxes, threshold):
        '''
        Parameters
        ----------
        classes : a list of optic disc of OD, optic disc of OS, macula

        Return
        ------
        Three bbox : Zone I, Posterior Zone II, Zone II
        '''

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
        zone2p = [cx-1.3*radius, cx+1.3*radius, cy-1.3*radius*4/3, cy+1.3*radius*4/3]
        zone2 = [cx-2*radius, cx+2*radius, cy-2*radius*4/3, cy+2*radius*4/3]
        
        return [zone1, zone2p, zone2]

    def draw_fundus_zones_on_image_array(image, zones, text ='', use_normalized_coordinates=True):
        
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        
        draw = ImageDraw.Draw(image_pil)
        im_width, im_height = image_pil.size
        
        colors = ["orange", "gold", "yellow"]
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

    def load_tf_graph(ckpt_path = '../src/odn/tf_ssd/export/frozen_inference_graph.pb',
                 label_path = '../src/odn/tf_ssd/fundus_label_map.pbtxt', 
                 num_classes = 2, verbose = True):
        '''
        Load tf graph from checkpoint
        
        Parameters
        ----------
        ckpt_path : path to the model checkpoint file. e.g., '../src/odn/tf_ssd/export/frozen_inference_graph.pb', 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        label_path : path to label file. Should follow this json format:
            item {
            id: 1
            name: 'OpticDisk'
            }

            item {
            id: 2
            name: 'Macula'
            }
        
        Returns
        -------    
        detection_graph : a TensorFlow computation, represented as a dataflow graph.
        '''

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        if verbose:
            print('category_index: ', category_index)
        
        return detection_graph, category_index


    def tf_batch_object_detection(detection_graph, category_index, FILES, 
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
                                        
                        zones = annotation.calculate_fundus_zones(classes[0], scores[0], boxes[0], threshold)
                        annotation.draw_fundus_zones_on_image_array(image_np, zones = zones, text=label)
                        
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


class dataset():

    def synthesize_anno(label_file, 
    dir_images, 
    dir_output = None, 
    drawzones = False,   
    verbose = True,
    display = 5 ):
        '''
        Synthesize the ground-truth images with annotations.

        Parameters
        ----------
        label_file : a csv file containing all image path and ROIs. e.g.,  '../data/fundus/all_labels.csv'

            The label file extends the VIA annotation format:

                filename: file name of the image
                class: denotes the class label of the ROI (region of interest)
                cx: image width
                cy: image height
                xmin: x-coordinate of the bottom left part of the ROI
                xmax: x-coordinate of the top right part of the ROI
                ymin: y-coordinate of the bottom left part of the ROI
                ymax: y-coordinate of the top right part of the ROI  
                laterality: L001 = OD, L002 = OS

        dir_images : The folder path containing all images, e.g.,  '../data/fundus/images/'

        dir_output : The generated images will be saved to this folder. e.g., '../data/fundus/ground_truth/'.
            If None, will not output images.
        '''
        
        df = pd.read_csv(label_file)

        if verbose:
            print('\n----------- \nContent of ', label_file)
            print(df.head())

        unique_files = list(set(df['filename'].values))
        if verbose:
            print('\n----------- \nUnique image files: ', len(unique_files))

        if verbose:
            print('\n----------- \ndistribution of ROI labels:')
            print( df['class'].value_counts() )
        
        i = 0
        for f in unique_files:            
            filepath = dir_images + '/' + f
            annotation.show_anno (filepath, df, dir_output, 
            drawzones = drawzones, showimg = (i < display) )
            i = i+1


    def split(label_file, dir_images, train_output, test_output,
    test_size = 0.2, verbose = True):
        '''
        Split the dataset into training and test sets.

        Parameters
        ----------
        label_file : a csv file containing all image path and ROIs. e.g.,  '../data/fundus/all_labels.csv'

            The label file extends the VIA annotation format:

                filename: file name of the image
                class: denotes the class label of the ROI (region of interest)
                cx: image width
                cy: image height
                xmin: x-coordinate of the bottom left part of the ROI
                xmax: x-coordinate of the top right part of the ROI
                ymin: y-coordinate of the bottom left part of the ROI
                ymax: y-coordinate of the top right part of the ROI  
                laterality: L001 = OD, L002 = OS

        dir_images : The folder path containing all images, e.g.,  '../data/fundus/images/'

        train_output, test_output : Target paths of generated train and test csv files.

        test_size : percentage of test set, in the range (0, 1.0)
        '''

        df = pd.read_csv(label_file)

        if verbose:
            print('\n----------- \nContent of ', label_file)
            print(df.head())

        unique_files = list(set(df['filename'].values))
        if verbose:
            print('\n----------- \nUnique image files: ', len(unique_files))

        m = len(unique_files)
        m_t = round(m * test_size)
        test_files = random.choices(unique_files, k = m_t)

        # Split to training set and test set

        test_set = df.loc[df['filename'].isin(test_files)].copy().sort_values(by=['filename']).reset_index(drop=True)
        train_set = df.loc[df['filename'].isin(test_files) == False].copy().sort_values(by=['filename']).reset_index(drop=True)

        # training set

        data = pd.DataFrame()
        data['format'] = train_set['filename']

        for i in range(data.shape[0]):
            # the path is relative to /src/odn/..py
            data['format'][i] = '../' + dir_images + data['format'][i] + ',' + str(train_set['xmin'][i]) + ',' + str(train_set['ymin'][i]) + ',' + str(train_set['xmax'][i]) + ',' + str(train_set['ymax'][i]) + ',' + train_set['class'][i]
            # print(data['format'][i])
            
        data.to_csv(train_output, header=None, index=None, sep=' ')


        # test set. We only need the filenames. The bbox of ROIs are not used.

        data = pd.DataFrame()
        data['format'] = test_set['filename']

        for i in range(data.shape[0]):
            data['format'][i] = '../' + dir_images + data['format'][i] + ',' + str(test_set['xmin'][i]) + ',' + str(test_set['ymin'][i]) + ',' + str(test_set['xmax'][i]) + ',' + str(test_set['ymax'][i]) + ',' + test_set['class'][i]

        data.to_csv(test_output, header=None, index=None, sep=' ')

    def add_new_row(img_, bboxes_, row, idx, 
    old_image_path_seg = '/images', 
    new_image_path_seg = '/images_public'):
    
        new_path = row[0].replace(old_image_path_seg, new_image_path_seg, 1) # .replace('.jpg', '.' + str(idx) + '.jpg', 1)
        status = cv2.imwrite(new_path, img_[:,:,::-1]) # bgr -> rgb
        
        s = new_path + \
        ',' + str(round( bboxes_[0][0], 1)) + \
        ',' + str(round( bboxes_[0][1], 1)) + \
        ',' + str(round( bboxes_[0][2], 1)) + \
        ',' + str(round( bboxes_[0][3], 1)) + \
        ',' + row[5]
        
        return s

    def deidentify(input_file = '../data/fundus/train.txt',
            output_file = '../data/fundus/train_public.txt', 
            old_image_path_seg = '/images', 
            new_image_path_seg = '/images_public'):

        # thanks to "https://github.com/Paperspace/DataAugmentationForObjectDetection"

        # %run ../src/odn/data_aug/data_aug.py
        # %run ../src/odn/data_aug/bbox_util.py

        # from data_aug.data_aug import *
        # from data_aug.bbox_util import *

        df = pd.read_csv(input_file, header=None, sep=',')

        datax = []

        DEID = True
        AUG_FLIP = False # always disabled for test
        AUG_COMBO = False # always disabled for test

        for index, row in df.iterrows():
            
            path = row[0][1:]
            # print([[row[1],row[2],row[3],row[4]]])
            img = cv2.imread(path)[:,:,::-1] #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb    
            
            bboxes = np.array([[row[1],row[2],row[3],row[4]]], dtype = float)
            
            # 
            # De-identify
            
            if DEID:
            
                w = img.shape[1]
                h = img.shape[0]

                # crop the right bottom section to detect text
                crop_img = img[round(h*0.93) : h, round(w * 0.93): w]
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) 
                _, thresh = cv2.threshold(gray, 127, 255,  cv2.THRESH_BINARY_INV)  # cv2.THRESH_OTSU |

                if (thresh.min() < 127):

                    img[ round(h*0.8) : h, round(w * 0.8): w] = 0 # de-identify

                    if (index < 5):
                        plt.figure()
                        plt.imshow(img)
                        plt.title( path )
                        plt.show()

                        plt.figure()
                        plt.imshow(gray)
                        plt.title( str(thresh.mean()) )
                        plt.show()

                s = dataset.add_new_row(
                    img, bboxes, row, 0, 
                    old_image_path_seg,
                    new_image_path_seg)
                datax.append(s)

            # 
            # DATA AUG
            
            if AUG_FLIP:
                
                raise NotImplementedError(__class__.__name__ + ' TODO: 把多个bbox和标签放到一起做aug操作')

                # FLIP
                img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
                s = add_new_row(img_, bboxes_, row, 1)
                datax.append(s)
            
            if AUG_COMBO: # FLIP + SCALE + TRANSLATE + ROTATE   
                
                raise NotImplementedError(__class__.__name__ + ' TODO: 把多个bbox和标签放到一起做aug操作')

                for idx in range(2, 7):
                    try:
                        img_, bboxes_ = RandomHorizontalFlip(0.5)(img.copy(), bboxes.copy())
                        img_, bboxes_ = RandomScale(0.1, diff = False)(img_, bboxes_)
                        img_, bboxes_ = RandomTranslate(0.1, diff = False)(img_, bboxes_)
                        img_, bboxes_ = RandomRotate(8)(img_, bboxes_)

                        s = add_new_row(img_, bboxes_, row, idx)
                        datax.append(s)

                    except Exception as err:
                        # print(row)
                        pass
            
                # show the first N examples
                if (index < 5):
                    plotted_img = draw_rect(img_, bboxes_)
                    plt.figure()
                    plt.imshow(plotted_img)
                    plt.title(row[5])
                    plt.show()
                
        sx = ''
        for d in datax:
            sx = sx + d + '\n'

        with open(output_file, "w") as f:
            f.write(sx)
            
        # finally, combine trainx.txt with train.txt

    # NOTE: images in the target folder will be overwritten if prefix is empty
    def batch_resize_fundus_image(folder, target, prefix = '', w=480, h=360):
        os.makedirs(target, exist_ok=True)
        
        imgExts = ["png", "bmp", "jpg"]
        for root, dirs, files in os.walk(folder):
            for f in files:
                ext = f[-3:].lower()
                if ext not in imgExts:
                    continue
                if prefix!='' and prefix is not None and f.startswith(prefix):
                    continue
                resize_fundus_image(root, f, target = target, prefix=prefix, w = w, h = h)

    def split_images_by_laterality(source_dir, target_dir, json_annos):
        '''
        Parameters
        ----------
        json_annos : a json object that contains a list of dict objects. 
            We will use its 'laterality' key to sperate images into OD and OS.
        '''
            
        IMAGEDIR = source_dir #'../data/fundus/images/'
        L2DIR = target_dir # '../data/fundus/L2'

        if os.path.exists(L2DIR):
            shutil.rmtree(L2DIR)
        os.makedirs(L2DIR)
        os.makedirs(L2DIR+'/L001')
        os.makedirs(L2DIR+'/L002')

        for idx, item in enumerate(json_annos):
            if 'laterality' in item.keys():
                lcode = item['laterality']    
                if lcode and not lcode.isspace():
                    file = item['filename']
                    shutil.copyfile(os.path.join(IMAGEDIR,file), os.path.join(L2DIR,lcode,file))

    def expand_images_by_flipping(imagedir):
        '''
        Create a flip copy for each image. The new image is named as filename_FLIP.jpg
        '''

        for root, dirs, files in os.walk(imagedir):
            for f in files:
                if f.endswith('.jpg'):
                    im = Image.open(os.path.join(root, f)).transpose(Image.FLIP_LEFT_RIGHT)                
                    im.save(os.path.join(root, f.replace('.jpg','_FLIP.jpg')))

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    
# All fundus images are 4:3 ratio. 
def resize_fundus_image(root, f, target, prefix='', w=480, h=360):
    filePath = os.path.join(root, f)
    newfilePath = os.path.join(target, prefix + f)    
    im = Image.open(filePath).resize((w,h)) 
    im.save(newfilePath)