from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle
import LetterDetection
import cv2
#from scipy.misc import toimage



def load_model(bin_dir):
    ''' Load model from .yaml and the weights from .h5
        Arguments:
            bin_dir: The directory of the bin (normally bin/)
        Returns:
            Loaded model from file
    '''

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def predict(curChar):
    ''' Called when user presses the predict button.
        Processes the canvas and handles the image.
        Passes the loaded image into the neural network and it makes
        class prediction.
    '''


    x = np.invert(curChar)

    ### Experimental
    # Crop on rows
    #x = crop(x)
    #x = x.T
    # Crop on columns
    #x = crop(x)
    #x = x.T

    cv2.imshow("cropped", x)
    cv2.waitKey()

    # Visualize new array
    #imsave('resized.png', x)
    x = imresize(x,(28,28))

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)
    #out = x

    # Generate response
    #response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
    #            'confidence': str(max(out[0]) * 100)[:6]}
    response = chr(mapping[(int(np.argmax(out, axis=1)[0]))])

    return response


if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='uses data from training.py and the EMNIST dataset')
    parser.add_argument('--bin', type=str, default='bin', help='Directory to the bin containing the model yaml and model h5 files')
    args = parser.parse_args()


    model = load_model(args.bin)
    mapping = pickle.load(open('%s/mapping.p' % args.bin, 'rb'))
    letDet = LetterDetection.seperate("hello_dig.png")
    #print(predict(letDet[1]))
    #print(len(letDet))
    for loc in range(len(letDet)):
        print(predict(letDet[loc]))
