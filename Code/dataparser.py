import struct
from typing import List
from image import Image


class Dataparser:
    def __init__(self):
        self.labels = []
        self.images: List[Image] = []
        self.imagesLin = []

    def loadLabels(self, labelsPath="train-labels.idx1-ubyte", updateImages=True):
        with open(labelsPath, "rb") as labelsFile:
            labelsFile.seek(8)
            labelBytes = labelsFile.read()
            self.labels = struct.unpack(">" + "B" * (len(labelBytes)), labelBytes)
        if updateImages and len(self.images) > 0:
            for imageIndex in range(len(self.images)):
                self.images[imageIndex].label = self.labels[imageIndex]

    def loadImages(self, imagesPath="train-images.idx3-ubyte"):
        with open(imagesPath, "rb") as imagesFile:
            imagesFile.seek(4)
            numImages = int.from_bytes(imagesFile.read(4), "big")
            imageRows = int.from_bytes(imagesFile.read(4), "big")
            imageColumns = int.from_bytes(imagesFile.read(4), "big")

            # images = np.empty((numImages, imageRows, imageColumns), np.ubyte)
            imageBytes = imagesFile.read()
            stepSize = imageRows * imageColumns

            self.imagesLin = [
                imageBytes[i : i + stepSize]
                for i in range(0, len(imageBytes), stepSize)
            ]

            for imageIndex in range(numImages):
                imageData = self.imagesLin[imageIndex]
                self.images.append(
                    Image(
                        imageData,
                        (-1 if len(self.labels) <= 0 else self.labels[imageIndex]),
                        imageColumns,
                        imageRows,
                    )
                )
