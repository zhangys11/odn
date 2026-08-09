import cv2
import sys
import tensorflow as tf
import numpy as np
from . import utils
from .tf_ssd.object_detection.utils import visualization_utils as vis_util

def test_camera():

    video_capture = cv2.VideoCapture(0)

    while True:

        # The video capture object can then be used to read frame by frame
        #   The img is literaly an image
        # is_sucessfuly_read is a boolean which returns true or false depending
        #   on whether the next frame is sucessfully grabbed.
        is_sucessfully_read, img = video_capture.read()

        # is_sucessfuly_read will return false when the a file ends, or is no 
        #   longer available, or has never been available
        if(is_sucessfully_read):
            cv2.imshow("Camera Feed", img)
        else:
            print ("Cannot read video capture object from %s. Quiting...", video_capture)
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break 

def realtime_object_detection(ckpt_path = '../src/odn/tf_ssd/export/frozen_inference_graph.pb',
                 label_path = '../src/odn/tf_ssd/fundus_label_map.pbtxt', num_classes = 2):
    '''
    Use the camera to do real-time object detection

    Parameters
    ----------
    detection_graph : a tf graph object
    '''

    gpus= tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    detection_graph, category_index = utils.load_tf_graph(ckpt_path,
                 label_path, 
                 num_classes, verbose = False)

    cv2.destroyAllWindows()
    video_capture = cv2.VideoCapture(0)

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
            
            while True:
                # The video capture object can then be used to read frame by frame
                #   The img is literaly an image
                # is_sucessfuly_read is a boolean which returns true or false depending
                #   on whether the next frame is sucessfully grabbed.
                is_sucessfully_read, img = video_capture.read()

                # is_sucessfuly_read will return false when the a file ends, or is no 
                #   longer available, or has never been available
                if(is_sucessfully_read):
                    image_np_expanded = np.expand_dims(img, axis=0)
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        img,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8, fontsize = 24)
                    cv2.imshow("Camera Feed", img)
                else:
                    print ("Cannot read video capture object from %s. Quiting...", video_capture)
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    video_capture.release()
                    cv2.destroyAllWindows()
                    break 