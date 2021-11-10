from time import perf_counter
import typing
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from layer import Layer
from dataparser import Dataparser

learningRate = 0.3
epochs = 1


def run(weights=None, modelFile=None, costGraphFile=None, testFrequency=1000):
    # Get training images and labels
    tStart = perf_counter()
    print("Loading training images...")
    dpTrain = Dataparser()
    dpTrain.loadLabels()
    dpTrain.loadImages()

    tLoadTrain = perf_counter()
    print(f"Done! Loading training images took {tLoadTrain-tStart}s")

    # Get test images and labels
    tLoadTest = perf_counter()
    dpTest = Dataparser()
    dpTest.loadLabels("t10k-labels.idx1-ubyte")
    dpTest.loadImages("t10k-images.idx3-ubyte")

    tLoadTest = perf_counter()
    print(f"Done! Loading test images took {tLoadTest-tLoadTrain}s")

    print("Creating layers...")
    layersTemplate = [len(dpTrain.images[0].normalizedData), 32, 10]
    layers = createLayers(layersTemplate)
    if weights == None:
        randomizeLayers(layers, 0.05)
    else:
        for layer in layers:
            if layer.previous == None:
                continue
            layer.weights = weights.pop()

    tCreate = perf_counter()

    print(f"Done! Creating layers took {tCreate-tLoadTrain}s")

    print("Running on all training examples...")
    for n in range(epochs):
        tA = perf_counter()
        accs = fullTrainingPass(layers, dpTrain, dpTest, testFrequency)
        tB = perf_counter()
        print(f"Done pass {n+1}/{epochs} in {tB-tA}s")
        if modelFile != None:
            print("Saving layers")
            saveWeights(layers, modelFile)
            print("Done!")
        tC = perf_counter()
        print("Testing model")
        accs.append(testPass(layers, dpTest))
        print(f"Done! Testing took {tC-tB}s. Accuracy: {accs[-1]}")
        if costGraphFile != None:
            print("Saving cost graph")
            appendCostGraph(accs, costGraphFile)
            print("Done!")

        tFinal = perf_counter()
        print(f"Total epoch time: {tFinal-tA}s")
        print("Running next epoch")
    tTrain = perf_counter()
    print(f"Done! Running all training examples took {tTrain-tCreate}s")


def fullTrainingPass(layers: typing.List[Layer], dpTrain, dpTest, testFrequency):
    accs = []
    t = tqdm(
        range(0, len(dpTrain.images), testFrequency),
        leave=True,
        desc="Epoch (Acc: 0%)",
    )
    for start in t:
        trainingPass(layers, dpTrain, start, start + testFrequency)
        acc = testPass(layers, dpTest)
        t.set_description(f"Epoch (Acc: {acc:.1%})")
        accs.append(acc)
    return accs


def trainingPass(layers: typing.List[Layer], dpTrain, start=0, end=None):
    if end == None:
        end = len(dpTrain.images)

    outputLayer = layers[-1]
    for t in tqdm(dpTrain.images[start:end], leave=False, desc="Training"):
        outputLayer.calculateValues(np.array(t.normalizedData))

        error = (
            outputLayer.outputValues
            * (1 - outputLayer.outputValues)
            * (t.expectedVector() - outputLayer.outputValues)
        )  # Calculate the output error, δk
        changes = []
        for layer in layers[::-1]:  # Loop through the layers from the back
            if layer.previous == None:
                continue
            changes = [
                learningRate * np.outer(error, layer.previous.outputValues)
            ] + changes  # Put a matrix of the output values multiplied by the error at the front of the changes list
            error = (
                np.dot(layer.weights.transpose(), error)
                * layer.previous.outputValues
                * (1 - layer.previous.outputValues)
            )
        for change, layer in zip(changes, layers[1:]):
            if layer.previous != None:
                layer.weights += change


def testPass(layers: typing.List[Layer], dpTest: Dataparser):
    costSum = 0
    correctGuesses = 0
    for t in tqdm(dpTest.images, leave=False, desc="Testing"):
        layers[-1].calculateValues(np.array(t.normalizedData))
        guess = np.argmax(layers[-1].outputValues)
        if guess == t.label:
            correctGuesses += 1
        costSum += layers[-1].cost(t.expectedVector())

    return correctGuesses / len(dpTest.images)


def saveWeights(layers, filename):
    allWeights = []
    for layer in layers:
        if not hasattr(layer, "weights"):
            continue
        allWeights.append(layer.weights)
    with open(filename, "wb+") as outFile:
        pickle.dump(allWeights, outFile)


def loadWeights(filename):
    with open(filename, "rb") as infile:
        return pickle.load(infile)


def appendCostGraph(costs, filename):
    with open(filename, "a+") as outfile:
        string = ",".join(map(lambda v: f"{v:,f}", costs))
        outfile.write(string + ",")


def createLayers(layers: list):
    layerList = []
    l = Layer(layers[0])
    layerList.append(l)
    for i in range(1, len(layers)):
        l = Layer(layers[i], l)
        layerList.append(l)
        layerList[i - 1].next = l
    return layerList


def randomizeLayers(layers: typing.List[Layer], variance: float):
    rV = np.vectorize(lambda _: random.uniform(-variance, variance))
    for l in layers:
        if not hasattr(l, "weights"):
            continue
        l.weights = rV(l.weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", help="The model file to be loaded")
    parser.add_argument("-m", help="The file to save the model to")
    parser.add_argument("-c", help="The file to save the cost graph to")
    parser.add_argument(
        "-f", help="The number of training examples between each test pass", type=int
    )

    args = parser.parse_args()

    weights = None
    testFrequency = args.f

    if args.i:
        weights = loadWeights(args.i)

    if not args.f:
        if args.c:
            testFrequency = 500
        else:
            testFrequency = 1000

    run(weights, args.m, args.c, testFrequency)
