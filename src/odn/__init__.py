
import pandas as pd

import os
import sys
if __package__:
    from . import utils, test_frcnn, fundus
    import warnings
    warnings.filterwarnings('ignore')
else:
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))
    import utils, test_frcnn, fundus

def predict_fundus_folder(folder, method = 'FRCNN', dir_output = 'inplace',model_path = None,
    label_path = None):
    '''
    method : 'FRCNN', 'SSD', or 'YOLO5'
    dir_output : specify a target folder or just 'inplace'
    model_path : if None, will use internal model
    label_path : the SSD pbtxt path
    '''

    wd = os.path.dirname(__file__) # the current working folder

    # construct the file list file
    fl = wd + '/filelist.tmp.txt' # uuid.uuid1() + '.txt'
    FILES = utils.get_all_images_in_dir(folder, fl, excludes = ['_FRCNN','_SSD','_YOLO5'])
    # test_set = pd.read_csv(fl, header=None, encoding = 'gbk')

    if method == 'FRCNN':       
                
        print('\n--- begin frcnn detection ---\n')
        if model_path is None:
            model_path = wd + '/models/vgg16/19e.hdf5'
        test_frcnn.predict_images(fl, 'vgg16', model_path, 32,
        wd + '/candidate.rois.txt')

        print('\n--- finish frcnn detection. rois saved to: ', wd + '/candidate.rois.txt' , ' ---\n')

        rois = fundus.annotation.rule_filter_rois(input_file = wd + '/candidate.rois.txt', 
        output_file = wd + '/filtered.rois.txt',
        verbose = False)

        print('\n--- finish roi filtering. rois saved to: ', wd + '/filtered.rois.txt' , ' ---\n')

        fundus.dataset.synthesize_anno(wd + '/filtered.rois.txt', 
                                dir_images = folder, 
                                dir_output = dir_output,
                                drawzones = True,
                                verbose = True,
                                suffix = '_FRCNN',
                                display = 0 )

        print('\n--- finish annotation synthesis. images saved to: ', dir_output , ' ---\n')

    elif method == 'SSD':

        if model_path is None:
            model_path = wd + '/tf_ssd/export/frozen_inference_graph.pb'
        if label_path is None:
            label_path = wd + '/tf_ssd/fundus_label_map.pbtxt'

        detection_graph, category_index = fundus.annotation.load_tf_graph(
            ckpt_path = model_path,
            label_path = label_path, 
            num_classes = 2)

        fundus.annotation.tf_batch_object_detection(detection_graph, category_index, FILES, 
                                        'inplace', 
                                        None,                        
                                        new_img_width = 900, fontsize = None, suffix = '_SSD')

    elif method == 'YOLO5':

        # requires torch
        import torch
        if not torch.cuda.is_available():
            print('Error: make sure torch and cuda is properly installed. \
            You may need to switch to a virtual env with these supports.')
            return

        if model_path is None:
            model_path = wd + '/torch_yolo/runs/train/exp15/weights/best.pt'

        fundus.annotation.torch_batch_object_detection(
            model_path = model_path,
            input_path = fl,
            conf_thres=0.3, iou_thres=0.5, max_det=2, 
            anno_pil = True, colors = [(200,100,100),(55,125,125)], 
            suffix = '_YOLO5', output_path = 'inplace',
            display = True, verbose = False
            )

if __name__ == "__main__":
    predict_fundus_folder('C:/Users/eleve/Desktop/横向2022/SZEH分区 第二次激光/激光  85个第二次', 
    method = 'YOLO5',
    dir_output = 'inplace',
    model_path = None,
    label_path = None)