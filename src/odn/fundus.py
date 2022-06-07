from http.client import UnimplementedFileMode
import numpy as np
import os
import os.path
import pathlib
import time
import pickle
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image
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

    def show_anno(filepath, df, savefolder = None, showimg = True):    
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
                edgecolor = 'white'
            elif row['class'] == 'Macula':
                edgecolor = 'lavender' #'azure'
                
            xy = (max(1, xmin),ymin-5)
            if (ymin < 5):
                xy = (max(1, xmin),ymax)
            ax.annotate(row['class'] + prob, xy=xy, color = edgecolor)

            # add bounding boxes to the image
            rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')

            ax.add_patch(rect)

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

class dataset():

    def synthesize_anno(label_file, 
    dir_images, 
    dir_output = None,    
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
            annotation.show_anno (filepath, df, dir_output, showimg = (i < display) )
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
    