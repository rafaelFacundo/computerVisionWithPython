import filtersFunctions
import videoFramesLabel
import os


videosToLabel = os.listdir("./videos");

filtersArray = []

for video in videosToLabel:
    newObjectToFilter = {}
    newObjectToFilter["videoPath"] = f"./videos/{video}"
    newObjectToFilter["videoSavePath"] = f"./framesLabeled/frames_{video.split(".")[0]}"
    newObjectToFilter["filtersToApply"] = [
        {"name": "canny", "filter": filtersFunctions.applyCannyEdgeDetection},
        {"name": "gaussian", "filter": filtersFunctions.applyGaussianBlur},
        {"name": "median", "filter": filtersFunctions.applyMedianBlur},
        {"name": "cannyBlur", "filter": filtersFunctions.cannyWithBlurAndGamma},
        {"name": "gamma", "filter": filtersFunctions.gammaCorrection},
    ]
    filtersArray.append(newObjectToFilter)

for element in filtersArray:
    videoFramesLabel.labelFrameYesOrNo(
        element["videoPath"],
        element["videoSavePath"],
        element["filtersToApply"]
    )
