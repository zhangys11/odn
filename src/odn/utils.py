import numpy as np
from keras_frcnn import roi_helpers
import cv2
import re
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def moving_average_3(a):    
	return np.concatenate(([a[0]], np.convolve(a, np.ones(3), 'valid') / 3, [a[-1]]))

def plot_training_curves(input_file = '../src/odn/training_log.txt', output_file = '../src/odn/training_curves.png'):
	'''
	Parameter
	---------
	input_file : the log file path. The log file is created from the commandline window output.
	'''

	L1={}
	L2={}
	L3={}
	L4={}
	ACC ={}

	epoch = 0

	with open(input_file) as f:
		content = f.readlines()
		for l in content:
			if l.startswith("Epoch"):
				arr = re.split('[: /]',l)
				epoch = int(arr[1])
			elif l.startswith("Classifier accuracy for bounding boxes from RPN"):
				arr = re.split('[:]',l)
				ACC[epoch] = float(arr[1])
			elif l.startswith("Loss RPN classifier"):
				arr = re.split('[:]',l)
				L1[epoch] = float(arr[1])
			elif l.startswith("Loss RPN regression"):
				arr = re.split('[:]',l)
				L2[epoch] = float(arr[1])
			elif l.startswith("Loss Detector classifier"):
				arr = re.split('[:]',l)
				L3[epoch] = float(arr[1])
			elif l.startswith("Loss Detector regression"):
				arr = re.split('[:]',l)
				L4[epoch] = float(arr[1])    

	fig, ax = plt.subplots(3,2, figsize = (40, 40))
	plt.rcParams.update({'font.size': 32})
	#plt.tick_params(axis='both', which='major', labelsize=24)
	#plt.tick_params(axis='both', which='minor', labelsize=24)
	plt.rcParams['xtick.labelsize'] = 32
	plt.rcParams['ytick.labelsize'] = 32

	labels = ['RPN Classification Error','RPN Regression Error','Region Classifier Classification Error',
			'Region Classifier Regression Error','Overall Classification Accuracy']

	M = [L1, L2, L3, L4, ACC]

	for idx in range(5):
		row = idx//2
		col = idx%2
		
		m = M[idx]
		
		ax[row, col].plot(list(m.keys()), moving_average_3(list(m.values())),label = labels[idx], color = 'gray', linewidth = 2)
		ax[row, col].scatter(list(m.keys()), moving_average_3(list(m.values())), s=240, facecolors='none', edgecolors='gray')
		ax[row, col].xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax[row, col].legend(loc='upper center')

	ax[2, 1].set_axis_off()

	plt.savefig(output_file)
	plt.close(fig)

def get_bbox(R, C, model_classifier, class_mapping, F, ratio, bbox_threshold = 0.8):
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
	all_dets = []
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

#			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))
			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
#			textOrg = (real_x1, real_y1-0)            
	return all_dets, bboxes, probs


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)