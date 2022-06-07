import numpy as np
import os
import os.path
import pathlib
import time
import pickle
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import shutil
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random

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

class dataset():

    def synthesize_gt(label_file, 
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
        for f in tqdm(unique_files):            
            filepath = dir_images + '/' + f
            annotation.show_anno (filepath, df, dir_output, showimg = (i < display) )
            i = i+1


    def split(label_file, dir_images, train_output = '../data/fundus/train.txt', test_output = '../data/fundus/test.txt',
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