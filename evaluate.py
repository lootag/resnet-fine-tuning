import os
from PIL import Image
from shutil import copyfile
from pyimagesearch import config
import numpy as np
from numpy import asarray
from tensorflow import keras
from tensorflow.keras.preprocessing import ImageDataGenerator



def main():
    logoFolder = "logo"
    noLogoFolder = "no_logo"
    os.mkdir(logoFolder)
    os.mkdir(noLogoFolder)
    modelPaths = ["adidas.classifier.h5"]
    evaluateNetworks(modelPaths, logoFolder, noLogoFolder)

def evaluateNetwork(modelPaths, logoFolder, noLogoFolder):
    models = loadModels(modelPaths)
    imagesToEvaluate = os.listdir(config.EVAL_PATH)   
    for imagePath in imagesToEvaluate:
        sourcePath = config.EVAL_PATH + image
        imageClass = classifyImage(imagePath, models)
        if imageClass == 0:
           destinationPath = logoFolder + image
           copyfile(sourcePath, destinationPath) 
        else if imageClass == 1:
           destinationPath = noLogoFolder + image
           copyfile(sourcePath, destinationPath)
    

def classifyImage(imagePath, models):
    for model in models:
        image = Image.open(imagePath)
        imageAsNumpyArray = asarray(image)
        predictions = model.predict(imageAsNumpyArray)
        labels = np.argmax(predictions, axis=1)
        label = labels[0,0] 
        if label == 0:
            return 0
    return 1


def loadModels(modelPaths):
    models = []
    for modelPath in modelPaths:
        model = keras.models.load_model(modelPath)
        models.append(model)

    return models

if __name__ == "__main__":
    main()
