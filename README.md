# About

We provide a set of object detection neural network training and evaluation functions.  
We currently support Faster-RCNN. An infant funuds image object detection application is provided.  

# Install

pip install tensorflow-gpu >= 2  
pip install odn

# How to use

[TO UPDATE]

1. Initialize

    from tlearner.efficientnet import transfer_learner
    learner = transfer_learner("my_model", W = 224) # my_model.h5df will be the model name

2. Load data

    learner.load_dataset('my_data_folder', PKL_PATH = "my_dataset.pkl") # Images in each subfolder of my_data_folder are treated as one class. If PKL_PATH is specified, a pkl that contains X_train, y_train, X_val, y_val, etc. will be generated.   
    print(learner.class_names) # print class names   

3. Train a new model

    hist = learner.train_custom_model("EfficientNetB1", batch = 8, epochs = [10,0], optimizer = "adam") # EfficientNetB0-B7 can be used.  
    plot_history(hist) # plot training curves

4. Evaluate
   
   learner.evaluate(N = 30) # predict the first N samples in X_val

5. Predict

    learner.predict_file('my_image.jpg')

6. Convert [Optional]
   
   learner.convert_to_tflite(v1 = True) # convert to a local tflite model

7. Load an existing model

   learner.load_best_model() # if you have an existing local model  
   learner.get_best_model().summary() # print model architecture  
   learner.plot_best_model() # save model architecture to a local image file

# Jupyter notebooks

Under /notebooks, we provide two examples. One is flower image classification; the other is fundus image classification.

# Deployment

After training, you will get a keras h5 model file. You can further convert it to tflite format, or tfjs format (efficient net is not supported yet).  
Then you can deploy on mobile device or browser-based apps.