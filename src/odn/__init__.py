import pandas as pd

import os
import sys
from pathlib import Path

if __package__:
    from . import utilities, fundus
else:
    '''
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # root directory, i.e., odn
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        print('*** Add odn root to sys.path: ', ROOT)
    '''
    import utilities, fundus

# sys.path.remove(os.path.dirname(__file__)) # to avoid conflicts with torch_yolo. Both have a model subpackage.

def predict_fundus_folder(folder, method = 'YOLO5', dir_output = 'inplace',model_path = None,
    label_path = None, callback = None):
    '''
    method : 'FRCNN', 'SSD', or 'YOLO5'
            'FRCNN' and 'SSD' require legacy tensorflow version. The install and config can be a real pain. We recommend using YOLO5.
    dir_output : specify a target folder or just 'inplace'
    model_path : if None, will use internal model
    label_path : the SSD pbtxt path
    callback : a callback function used to notify upper app.
    '''

    if callback is None:
        callback = print

    wd = os.path.dirname(__file__) # the current working folder

    # construct the file list file
    fl = wd + '/filelist.txt' # uuid.uuid1() + '.txt'
    FILES = utilities.get_all_images_in_dir(folder, fl, excludes = ['_FRCNN','_SSD','_YOLO5'])
    # test_set = pd.read_csv(fl, header=None, encoding = 'gbk')
    if FILES is None or len(FILES) <=0:
        callback('Error: No files found in folder ', folder)
        return

    if method == 'FRCNN':

        if __package__:
            from . import test_frcnn
        else:
            import test_frcnn

        callback('begin frcnn detection')
        if model_path is None:
            model_path = wd + '/models/vgg16/19e.hdf5'
        test_frcnn.predict_images(fl, 'vgg16', model_path, 32,
        wd + '/candidate.rois.txt')

        callback('finish frcnn detection. rois saved to: ' + wd + '/candidate.rois.txt')

        rois = fundus.annotation.rule_filter_rois(input_file = wd + '/candidate.rois.txt', 
        output_file = wd + '/filtered.rois.txt',
        verbose = False)

        callback('finish roi filtering. rois saved to: ' + wd + '/filtered.rois.txt')

        fundus.dataset.synthesize_anno(wd + '/filtered.rois.txt', 
                                dir_images = folder, 
                                dir_output = dir_output,
                                drawzones = True,
                                verbose = True,
                                suffix = '_FRCNN',
                                display = 0 )

    elif method == 'SSD':

        if model_path is None:
            model_path = wd + '/tf_ssd/export/frozen_inference_graph.pb'
        if label_path is None:
            label_path = wd + '/tf_ssd/fundus_label_map.pbtxt'

        detection_graph, category_index = fundus.annotation.load_tf_graph(
            ckpt_path = model_path,
            label_path = label_path, 
            num_classes = 2)

        callback('load model: ' + model_path)
        callback('load label map: ' + label_path)

        fundus.annotation.tf_batch_object_detection(detection_graph, category_index, FILES, 
                                        'inplace', 
                                        None,                        
                                        new_img_width = 900, fontsize = None, suffix = '_SSD')
        
    elif method == 'YOLO5':

        # requires torch
        import torch
        if not torch.cuda.is_available():
            callback('Error: make sure torch and cuda is properly installed. \
            You may need to switch to a virtual env with these supports.')
            
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
    else:
        callback('Error: Unsupported detection method ' + method)
        return

    callback('\nfinish ' + method + ' detection. \nimages saved to: ' + dir_output + '')

if __name__ == "__main__":
    predict_fundus_folder('C:/Users/eleve/Desktop/横向2022/SZEH分区 第二次激光', 
    method = 'SSD',
    dir_output = 'inplace',
    model_path = None,
    label_path = None)