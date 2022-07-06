import cv2
from .darkflow_yolo.darkflow.net.build import TFNet
import matplotlib.pyplot as plt

def load_tfnet(options = {
        'model': 'darkflow_yolo/cfg/yolo.cfg', # 'bin/yolov1.cfg',
        'load': 'darkflow_yolo/bin/yolov2.weights', # 'bin/yolov1.weights',
        'threshold': 0.3,
        'gpu': 0.7
    }):

    # define the model options and run
    tfnet = TFNet(options)

    return tfnet

def predict(tfnet, imgpath):

    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # use YOLO to predict the image
    results = tfnet.return_predict(img)
    print(results)

    plt.figure(figsize=(10,10))

    # pull out some info from the results
    for result in results:
        
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label'] + ': ' + str(round(result['confidence'],3))


        # add the box and label and display it
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        
    plt.imshow(img)
    plt.show()