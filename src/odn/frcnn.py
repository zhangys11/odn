# Based on Kentaro's github project: https://github.com/kentaroy47/frcnn-from-scratch-with-keras

# -*- coding: utf-8 -*-
# stuff
from __future__ import division
import random
import pprint
import keras
import sys
import time
from optparse import OptionParser
import pickle
import os
import numpy as np

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import data_generators
from keras_frcnn import config
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

def train_rpn (train_path, output_weight_path, num_rois, 
               network = 'vgg16', input_weight_path = None, # pre-trained weights for transfer learning
               parser = 'pascal_voc', # anno file format
               config_filename = 'train_rpn.config',
               horizontal_flips = False, vertical_flips = False, rot_90 = False, # data augmentation
               num_epochs = 50):

    # gpu setting
    if 'tensorflow' == K.backend():
        import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    set_session(tf.Session(config=config2))

    # make dirs to save rpn
    # "./models/rpn/rpn"
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir("models/rpn"):
        os.mkdir("models/rpn")

    # we will train from pascal voc 2007
    # you have to pass the directory of VOC with -p
    if not train_path:   # if filename is not given
        raise ValueError('Error: path to training data must be specified. Pass --path to command line')

    if parser == 'pascal_voc':
        from keras_frcnn.pascal_voc_parser import get_data
    elif parser == 'simple':
        from keras_frcnn.simple_parser import get_data
    else:
        raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

    # pass the settings from the command line, and persist them in the config object
    C = config.Config()

    # set data argumentation
    C.use_horizontal_flips = bool(horizontal_flips)
    C.use_vertical_flips = bool(vertical_flips)
    C.rot_90 = bool(rot_90)

    C.model_path = output_weight_path
    C.num_rois = int(num_rois)

    # we will use resnet. may change to vgg
    if network == 'vgg' or network == 'vgg16':
        C.network = 'vgg16'
        from keras_frcnn import vgg as nn
    elif network == 'resnet50':
        from keras_frcnn import resnet as nn
        C.network = 'resnet50'
    elif network == 'vgg19':
        from keras_frcnn import vgg19 as nn
        C.network = 'vgg19'
    elif network == 'mobilenetv1':
        from keras_frcnn import mobilenetv1 as nn
        C.network = 'mobilenetv1'
    elif network == 'mobilenetv1_05':
        from keras_frcnn import mobilenetv1_05 as nn
        C.network = 'mobilenetv1_05'
    elif network == 'mobilenetv1_25':
        from keras_frcnn import mobilenetv1_25 as nn
        C.network = 'mobilenetv1_25'
    elif network == 'mobilenetv2':
        from keras_frcnn import mobilenetv2 as nn
        C.network = 'mobilenetv2'
    else:
        print('Not a valid model')
        raise ValueError


    # check if weight path was passed via command line
    if input_weight_path:
        C.base_net_weights = input_weight_path
    else:
        # set the path to weights based on backend and model
        C.base_net_weights = nn.get_weight_path()


    # place weight files on your directory
    ase_net_weights = nn.get_weight_path()


    #### load images here ####
    # get voc images
    all_imgs, classes_count, class_mapping = get_data(train_path)

    print(classes_count)

    # add background class as 21st class
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    config_output_filename = config_filename

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

    random.shuffle(all_imgs)

    num_imgs = len(all_imgs)

    # split to train and val
    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))


    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

    # set input shape
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # create rpn model here
    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    # rpn outputs regression and cls
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    model_rpn = Model(img_input, rpn[:2])

    #load weights from pretrain
    try:
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
    #	model_classifier.load_weights(C.base_net_weights, by_name=True)
        print("loaded weights!")
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')

    # compile model
    optimizer = Adam(lr=1e-5, clipnorm=0.001)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

    # write training misc here
    epoch_length = 100
    num_epochs = int(num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True


    # start acutual training here
    #X, Y, img_data = next(data_gen_train)
    #
    ##loss_rpn = model_rpn.train_on_batch(X, Y)
    #P_rpn = model_rpn.predict_on_batch(X)

    # you should enable NMS when you visualize your results.
    # NMS will filter out redundant predictions rpn gives, and will only leave the "best" predictions.
    # P_rpn = model_rpn.predict_on_batch(image)
    # R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
    # X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
    # this will output the binding box axis. [x1,x2,y1,y2].

    Callbacks=keras.callbacks.ModelCheckpoint("./models/rpn/rpn."+network+".weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=4)
    callback=[Callbacks]
    if len(val_imgs) == 0:
        # assuming you don't have validation data
        history = model_rpn.fit_generator(data_gen_train,
                        epochs=num_epochs, steps_per_epoch = epoch_length, callbacks=callback)
        loss_history = history.history["loss"]
    else:
        history = model_rpn.fit_generator(data_gen_train,
                        epochs=num_epochs, validation_data=data_gen_val,
                        steps_per_epoch=epoch_length, callbacks=callback, validation_steps=100)
        loss_history = history.history["val_loss"]

    numpy_loss_history = np.array(loss_history)
    np.savetxt(network+"_rpn_loss_history.txt", numpy_loss_history, delimiter=",")
