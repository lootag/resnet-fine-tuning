import os
from PIL import Image
from shutil import copyfile
from pyimagesearch import config
import numpy as np
from numpy import asarray
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def main():
    logoFolder = "logo"
    noLogoFolder = "no_logo"
    os.mkdir(logoFolder)
    os.mkdir(noLogoFolder)
    modelPaths = ["adidas.classifier.h5"]
    evaluateNetworks(modelPaths, logoFolder, noLogoFolder)

def evaluateNetworks(modelPaths, logoFolder, noLogoFolder):
    models = loadModels(modelPaths)
    imagesToEvaluate = os.listdir(config.EVAL_PATH)   
    for imageName in imagesToEvaluate:
        sourcePath = os.path.join(config.EVAL_PATH, imageName)
        imageClass = classifyImage(sourcePath, models)
        if imageClass == 0:
           destinationPath = os.path.join(logoFolder ,imageName)
           copyfile(sourcePath, destinationPath) 
        elif imageClass == 1: 
           destinationPath = os.path.join(noLogoFolder ,imageName)
           copyfile(sourcePath, destinationPath)
    

def classifyImage(imagePath, models):
    for model in models:
        image = Image.open(imagePath)
        imageAsNumpyArray = asarray(image)
        imageAsNumpyArray = imageAsNumpyArray[np.newaxis, ...]
        print(imageAsNumpyArray.shape)
        predictions = model.predict(imageAsNumpyArray)
        labels = np.argmax(predictions, axis=1)
        print(labels)
        label = labels[0] 
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
