import os
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

def findValidFrames(files, directory, maxFrames, startFrom=0, validExtensions=('.png', '.jpg', '.jpeg', '.bmp')):

    validIndexes = []

    iterateForward = maxFrames > 0
    if iterateForward:
        iterationRange = range(startFrom, len(files))
    else:
        maxFrames = abs(maxFrames)
        startFrom = len(files) - 1 if startFrom <= 0 else startFrom - 1
        iterationRange = range(startFrom, -1, -1)

    for index in iterationRange:
        file = files[index]
        if file.startswith('.') or not file.lower().endswith(validExtensions):
            continue

        try:
            with Image.open(os.path.join(directory, file)) as img:
                img.verify()
                validIndexes.append(index)
                if len(validIndexes) == maxFrames:
                    break
        except (IOError, UnidentifiedImageError):
            continue

    return validIndexes

class LoadQueuedBatchImages:
    def __init__(self):
        self.iterator = {}
        self.reset = {}

    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),

            },
            "optional": {
                "batchSize": ("INT", {"default": 8, "min": 1, "step": 1}),
                "preFrame": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "resetStartAt": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
                "hidden": {
                    "uniqueId": "UNIQUE_ID",
                }
        }
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN",)
    RETURN_NAMES = ("IMAGE","preFramed","index","hasNext")
    FUNCTION = "mainLoadImages"
    CATEGORY = "Queue Tools"

    def mainLoadImages(self, directory, batchSize, preFrame, reset, resetStartAt, uniqueId):

        directory = directory.strip()
        directory = os.path.abspath("input") + "/" + directory
        if not os.path.isdir(directory):
            raise Exception("Directory not found: " + directory)
        files = sorted(os.listdir(directory))
        if len(files) == 0:
            raise Exception("No files found: " + directory)

        iterator = 0
        if reset:
            nextIndex = resetStartAt
            if resetStartAt > 0:
                nIndex = findValidFrames(files, directory, resetStartAt + 1, 0)
                if len(nIndex) == 0:
                    raise Exception("No files found: " + directory)
                nextIndex = nIndex[-1]
            self.iterator[uniqueId] = nextIndex
        iterator = self.iterator.get(uniqueId, iterator)

        images = []
        hasPreframe = 0

        # if we want to use a previous frame to interpolate from to our first image
        if preFrame and iterator > 0:
            prevIndex = findValidFrames(files, directory, -1, iterator)
            if prevIndex:
                filePath = os.path.join(directory, files[prevIndex[0]])
                with Image.open(filePath) as img:
                    image = img.convert("RGB")
                    images.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,])
                    hasPreframe = 1

        # we try to get one extra image to see if this is the last batch...
        validIndexes = findValidFrames(files, directory, batchSize + 1, iterator)
        counter = 0
        for index in validIndexes:
            counter += 1
            filePath = os.path.join(directory, files[index])
            with Image.open(filePath) as img:
                image = img.convert("RGB")
                images.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,])
        # ...then we remove it
        hasNext = counter > batchSize
        if hasNext:
            images.pop()

        #
        if hasPreframe and counter == 0:
            images.pop(0)

        if validIndexes:
            self.iterator[uniqueId] = validIndexes[-1] + 1

        if len(images) == 0:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        return (torch.cat(images, dim=0), hasPreframe, iterator, hasNext)
