# USAGE
# python build_dataset.py

# import the necessary packages
from pyimagesearch import config
import random
from shutil import copyfile
import os
import math

def main():
    run()

def run():
    os.mkdir(config.BASE_PATH)
    os.mkdir(os.path.join(config.BASE_PATH, "training"))
    os.mkdir(os.path.join(config.BASE_PATH, "validation"))
    os.mkdir(os.path.join(config.BASE_PATH, "testing"))
    for className in config.CLASSES:
        buildDataSet(className)

def buildDataSet(className):
    pathToClass = os.path.join(config.ORIG_INPUT_DATASET, className)
    trainFolder = os.path.join(config.BASE_PATH, "training", className)
    valFolder = os.path.join(config.BASE_PATH, "validation", className)
    testFolder = os.path.join(config.BASE_PATH, "testing", className)
    os.mkdir(trainFolder)
    os.mkdir(valFolder)
    os.mkdir(testFolder)
    classFileNames = os.listdir(pathToClass)
    shuffledClassFileNames = classFileNames
    random.shuffle(shuffledClassFileNames)
    numberOfTrainSamples = math.floor(config.TRAIN_SPLIT * len(shuffledClassFileNames))
    numberOfValSamples = math.floor(config.VAL_SPLIT * len(shuffledClassFileNames))
    trainFileNames = shuffledClassFileNames[:numberOfTrainSamples]
    valFileNames = shuffledClassFileNames[numberOfTrainSamples:numberOfTrainSamples + numberOfValSamples]
    testFileNames = shuffledClassFileNames[numberOfTrainSamples + numberOfValSamples:]
    movePathsToDestinantionFolder(trainFileNames, pathToClass, trainFolder)
    movePathsToDestinantionFolder(valFileNames, pathToClass, valFolder)
    movePathsToDestinantionFolder(testFileNames, pathToClass, testFolder)


def movePathsToDestinantionFolder(inputFileNames, sourceFolder, destinationFolder):
    for fileName in inputFileNames:
        sourcePath = os.path.join(sourceFolder, fileName)
        destinationPath = os.path.join(destinationFolder, fileName)
        copyfile(sourcePath, destinationPath)


if __name__ == "__main__":
    main()



