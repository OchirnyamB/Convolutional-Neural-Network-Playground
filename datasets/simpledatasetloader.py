import cv2
import numpy as np
import os

class SimpleDataSetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
           self.preprocessors = []
    
    def preprocess(self, image):
        # resize the image to fixed size, ignore the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

    def load(self, imagePaths, verbose=-1):
        data=[]
        labels=[]

        # Loop over the input images
        for(i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            # Let's assume that our datasets are organized on disk
            # according to the following directory structure: /dataset_name/class/image.png
            label = imagePath.split(os.path.sep)[-2]

            # Loop over the preprocessors and apply each to the image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            # Treat the preprocessed image as a feature vector
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))
        
        return (np.array(data), np.array(labels))