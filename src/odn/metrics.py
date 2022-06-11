import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os
import scipy.stats as stats
from IPython.core.display import display, HTML

def bbox_IoU(boxA, boxB):
	'''
	IOU=(A∩B)/(A∪B) :Intersection over Union 
	'''
    
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

def fundus_metrics(gt = '../data/fundus/all_labels.csv', 
pred = '../src/odn/rois.txt'):

	'''
	Paramters
	---------
	gt : ground-truth annotation csv file
	pred : predicted result csv file  
	'''

	df = pd.read_csv(gt)
	rois = pd.read_csv(pred)

	IOU_O = {}
	IOU_M = {}
	RCE_O = {}
	RCE_M = {}
	P_O = {}
	P_M = {}


	for fn in set(rois['filename'].values):
		
		# IOU for OpticDisk
		
		width = 0
		height = 0
		bbox_opticdisk_gt = [0,0,0,0]
		bbox_opticdisk_gts = df.loc[(df['filename'] == fn) & (df['class'] == 'OpticDisk')][['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']].values
		# print(len(bbox_opticdisk_gts),fn)
		if (len(bbox_opticdisk_gts) > 0):
			bbox_opticdisk_gt = bbox_opticdisk_gts[0][0:4]
			width = bbox_opticdisk_gts[0][4]
			height = bbox_opticdisk_gts[0][5]
			#print(bbox_opticdisk_gt)
		else:
			print(fn, 'no optic disc')
		
		
		prob = 0
		bbox_opticdisk_pred = [0,0,0,0]
		bbox_opticdisk_preds = rois.loc[(rois['filename'] == fn) & (rois['class'] == 'OpticDisk')][['xmin', 'ymin', 'xmax', 'ymax', 'prob']].values
		if (len(bbox_opticdisk_preds) > 0):
			bbox_opticdisk_pred = bbox_opticdisk_preds[0][0:4]
			prob = bbox_opticdisk_preds[0][4]
			#print(bbox_opticdisk_pred)
		
		if (bbox_opticdisk_gt.tolist() == [0,0,0,0] and bbox_opticdisk_pred.tolist() == [0,0,0,0]):
			IOU_O[fn] = 1 # there is acturally no this structure and we didn't detect it.
		else:
			IOU_O[fn] = bbox_IoU(bbox_opticdisk_gt,bbox_opticdisk_pred)
		RCE_O[fn] = bbox_RCE(bbox_opticdisk_gt,bbox_opticdisk_pred, width, height)
		P_O[fn] = prob
		
		# IOU for Macula
		
		bbox_macula_gt = [0,0,0,0]
		bbox_macula_gts = df.loc[(df['filename'] == fn) & (df['class'] == 'Macula')][['xmin', 'ymin', 'xmax', 'ymax']].values
		# print(len(bbox_opticdisk_gts),fn)
		if (len(bbox_macula_gts) > 0):
			bbox_macula_gt = bbox_macula_gts[0]
			#print(bbox_opticdisk_gt)
		else:
			print(fn, 'no macula')
		
		prob = 0
		bbox_macula_pred = [0,0,0,0]
		bbox_macula_preds = rois.loc[(rois['filename'] == fn) & (rois['class'] == 'Macula')][['xmin', 'ymin', 'xmax', 'ymax', 'prob']].values
		if (len(bbox_macula_preds) > 0):
			bbox_macula_pred = bbox_macula_preds[0][0:4]
			prob = bbox_macula_preds[0][4]
			#print(bbox_opticdisk_pred)
		
		if (bbox_macula_gt.tolist() == [0,0,0,0] and bbox_macula_pred.tolist() == [0,0,0,0]):
			IOU_M[fn] = 1 # there is acturally no this structure and we didn't detect it.
		else:
			IOU_M[fn] = bbox_IoU(bbox_macula_gt,bbox_macula_pred)
		RCE_M[fn] = bbox_RCE(bbox_macula_gt,bbox_macula_pred, width, height)
		P_M[fn] = prob


	return 	IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M 

def fundus_compare_metrics(gt = '../data/fundus/all_labels.csv', 
pred = '../src/odn/rois.txt', 
output_file = '../src/odn/comparison_with_metrics.jpg',
image_dirs = [
	'../data/fundus/ground_truth_public', 
	'../data/fundus/odn_19e_raw',   
	'../data/fundus/odn_19e_naive',
	'../data/fundus/odn_19e'], image_subset = [], verbose = True ):

	'''
	display (ground truth, raw, naive, processed) column by column; generate a single large plot.
	
	Parameters
	----------
	image_subset : a list of selected image file names. If None or empty, will use all images.
	'''
		
	df = pd.read_csv(pred)
	uniquefiles = list(set(df['filename'].values))
	uniquefiles.sort()	

	if image_subset is None or len(image_subset) <=0:
		image_subset = uniquefiles

	n = len(image_subset)

	fig, ax = plt.subplots(n, len(image_dirs) + 1, figsize = (16 * len(image_dirs), 9 * n))
	plt.rcParams.update({'font.size': 32})

	if verbose:
		IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M = fundus_metrics(gt, pred)

	for row,fn in enumerate(image_subset):

		if verbose:

			text =  fn[:20] + '\n' + fn[20:] + '\n\n' # manually break word
			text += 'Probability of Optic disc: ' + str(round(P_O[fn],3)) + '\n'
			text += 'Probability of Macula: ' + str(round(P_M[fn],3)) + '\n'
			text += 'IoU of Optic disc: ' + str( round(IOU_O[fn],3)) + '\n'
			text += 'IoU of Macula: ' + str( round(IOU_M[fn],3)) + '\n'
			text += 'RCE of Optic disc: ' + str(round(RCE_O[fn],3)) + '\n'
			text += 'RCE of Macula: ' + str(round(RCE_M[fn],3)) + '\n'
			print(text)
		
		for col in range(len(image_dirs) + 1): 
			
			ax[row, col].set_axis_off()
			if (col < len(image_dirs)):
				image = plt.imread(image_dirs[col] + '/' + fn)
				crop_ratio = 0.08
				img_cropped = image[(int)(image.shape[0] * crop_ratio):(int)(image.shape[0] * (1-crop_ratio)), 
								(int)(image.shape[1] * crop_ratio):(int)(image.shape[1] * (1-crop_ratio)),
								:]
				ax[row, col].imshow(img_cropped) 
			elif (col == len(image_dirs)):
				# print(text)
				ax[row, col].text(0.1, 0.2, text, transform=ax[row, col].transAxes, size = 45, wrap = True) 
		
	# fig.tight_layout()
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.1, hspace=-0.1)

	plt.savefig(output_file)
	plt.close(fig)

def fundus_compare_metrics_separate(input_file = '../src/odn/rois.txt', 
output_dir = './comparison_separate/',
image_dirs = [
	'../data/fundus/ground_truth_public', 
	'../data/fundus/odn_19e_raw', 
	'../data/fundus/odn_19e'], image_subset = [], verbose = True ):

	'''
	display (ground truth, raw, processed) column by column; 
	generate a single plot for each image.

	Parameters
	----------
	image_subset : a list of selected image file names. If None or empty, will use all images.
	
	Remarks
	-------
	subplots_adjust - The parameter meanings (and suggested defaults) are:

	left  = 0.125  # the left side of the subplots of the figure
	right = 0.9    # the right side of the subplots of the figure
	bottom = 0.1   # the bottom of the subplots of the figure
	top = 0.9      # the top of the subplots of the figure
	wspace = 0.2   # the amount of width reserved for blank space between subplots
	hspace = 0.2   # the amount of height reserved for white space between subplots
	'''
		
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	if not output_dir.endswith('/'):
		output_dir = output_dir + '/'
	
	df = pd.read_csv(input_file)
	uniquefiles = list(set(df['filename'].values))
	uniquefiles.sort()

	if image_subset is None or len(image_subset) <=0:
		image_subset = uniquefiles

	n = len(image_subset)

	for row,fn in enumerate(uniquefiles):

		fig, ax = plt.subplots(1, len(image_dirs), figsize = (32, 8))
		plt.rcParams.update({'font.size': 32})
		
		for col in range(len(image_dirs)): 
			
			ax[col].set_axis_off()        
			image = plt.imread(image_dirs[col] + '/' + fn)
			crop_ratio = 0.05
			img_cropped = image[(int)(image.shape[0] * crop_ratio):(int)(image.shape[0] * (1-crop_ratio)), 
								(int)(image.shape[1] * crop_ratio * 2):(int)(image.shape[1] * (1-crop_ratio*2)),
								:]
			ax[col].imshow(img_cropped) 
		
		# fig.tight_layout()
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.1, hspace=-0.1)

		plt.savefig(output_dir + str(row).zfill(2) + '_' + fn)
		plt.close(fig)

def metric_histogram(gt = '../data/fundus/all_labels.csv', 
pred = '../src/odn/rois.txt',
output_file = './metrics.png'):

	fig, ax = plt.subplots(2,3, figsize = (60, 30))
	plt.rcParams.update({'font.size': 32})
	#plt.tick_params(axis='both', which='major', labelsize=24)
	#plt.tick_params(axis='both', which='minor', labelsize=24)
	plt.rcParams['xtick.labelsize'] = 32
	plt.rcParams['ytick.labelsize'] = 32

	labels = ['IoU of Optic Disc', 'RCE of Optic Disc', 'ACC of Optic Disc', 'IoU of Macula', 'RCE of Macula', 'ACC of Macula']
	IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M = fundus_metrics(gt, pred)

	for idx, M in enumerate([IOU_O, RCE_O, P_O, IOU_M, RCE_M, P_M]): # use P_O and P_M to replace ACC_O and ACC_M
		row = idx // 3
		col = idx % 3
		
		m = list(M)
		if hasattr(M, 'values'): # if M is a pandas dataframe
			m = list(M.values())
			
		density = stats.gaussian_kde(m, bw_method = 'silverman')
		n, x, _ = ax[row, col].hist(m,label = labels[idx], bins = 10, 
									density = None, histtype='step', linewidth=2, facecolor='lightgray', 
									hatch='.', edgecolor='k',fill=False) # hatch='...'
		# ax[row, col].plot(x, density(x))
		ax[row, col].xaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
		ax[row, col].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
		ax[row, col].legend(loc='upper center')

	plt.savefig(output_file)
	plt.show()
	plt.close(fig)


def fundus_compare_metrics_html(gt = '../data/fundus/all_labels.csv', 
pred = '../src/odn/rois.txt'):
	'''
	Used in jupyter notebook. Output comparison result in a html table.
	'''

	df = pd.read_csv(pred)
	uniquefiles = list(set(df['filename'].values))
	uniquefiles.sort()

	IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M = fundus_metrics(gt, pred)
	SHOWSET = [P_O, P_M, IOU_O, IOU_M, RCE_O, RCE_M]

	s = '<table>'
	s += '<tr>'
	s += '<td></td>'
	s += '<td>File Name</td>'
	s += '<td>Probability of Optic Disc</td>'
	s += '<td>Probability of Macula</td>'
	s += '<td>IoU of Optic Disc</td>'
	s += '<td>IoU of Macula</td>'
	s += '<td>RCE of Optic Disc</td>'
	s += '<td>RCE of Macula</td>'
	s += '</tr>'

	for row,fn in enumerate(uniquefiles):
		s+= '<tr>'
		s+= '<td>' + str(row) + '</td>'
		s+= '<td>' + fn + '</td>'
		for ds in SHOWSET:
			s+= '<td>' + str(round(ds[fn],3)) + '</td>'    
		s+= '</tr>'
		
	s += '</table>'
	display(HTML(s))

def fundus_compare_metrics_html2(gt = '../data/fundus/all_labels.csv', 
pred1 = '../src/odn/rois.txt', pred2 = '../src/odn/rois_naive.txt'):
	'''
	Find images with different results between naive method and our method
	'''

	df = pd.read_csv(pred1)
	uniquefiles = list(set(df['filename'].values))
	uniquefiles.sort()

	DIFF_FILES = []

	IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M = fundus_metrics(gt, pred1)
	SHOWSET = [P_O, P_M, IOU_O, IOU_M, RCE_O, RCE_M]

	IOU_O , IOU_M ,	RCE_O, RCE_M , P_O, P_M = fundus_metrics(gt, pred2)
	SHOWSET_2 = [P_O, P_M, IOU_O, IOU_M, RCE_O, RCE_M]

	s = '<table>'
	s += '<tr>'
	s += '<td></td>'
	s += '<td>File Name</td>'
	s += '<td>Probability of Optic Disc</td>'
	s += '<td>Probability of Macula</td>'
	s += '<td>IoU of Optic Disc</td>'
	s += '<td>IoU of Macula</td>'
	s += '<td>RCE of Optic Disc</td>'
	s += '<td>RCE of Macula</td>'

	s += '<td>Probability of Optic Disc</td>'
	s += '<td>Probability of Macula</td>'
	s += '<td>IoU of Optic Disc</td>'
	s += '<td>IoU of Macula</td>'
	s += '<td>RCE of Optic Disc</td>'
	s += '<td>RCE of Macula</td>'

	s += '</tr>'

	for row,fn in enumerate(uniquefiles):
		
		for idx in range(len(SHOWSET)):
			if str(round(SHOWSET[idx][fn],3)) != str(round(SHOWSET_2[idx][fn],3)):        
				s+= '<tr>'
				s+= '<td>' + str(row) + '</td>'
				s+= '<td>' + fn + '</td>'
				for idx in range(len(SHOWSET)):
					s+= '<td>' + str(round(SHOWSET[idx][fn],3)) + '</td>'
					#s+= '<td>' + str(round(SHOWSET_2[idx][fn],3)) + '</td>'
					#print(round(SHOWSET_2[0][fn],3))
				for idx in range(len(SHOWSET_2)):
					#s+= '<td>' + str(round(SHOWSET[idx][fn],3)) + '</td>'
					s+= '<td>' + str(round(SHOWSET_2[idx][fn],3)) + '</td>'
				s+= '</tr>'
				
				DIFF_FILES.append(fn)
				
				break
		
	s += '</table>'
	display(HTML(s))
	
	return DIFF_FILES