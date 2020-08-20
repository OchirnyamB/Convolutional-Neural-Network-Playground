from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDataSetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse 
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pretrained model")
args = vars(ap.parse_args())

# initialize class labels
imagePaths = list(paths.list_images(args["dataset"]))
classLabels = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classLabels = [str(x) for x in np.unique(classLabels)]

print(classLabels)

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
imagePaths = np.array(imagePaths)
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDataSetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

# load the pretrained network
print("[INFO] loading pre-trained network")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
# returns a list of probabilities for every image in data - one probability for each class label
# taking the argmax on axis=1 infds the index of the class label with the largest probability for each image.
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)