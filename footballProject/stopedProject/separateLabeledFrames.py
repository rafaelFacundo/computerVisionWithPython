import os
import glob
import shutil

pathToMoveTheFrames = './labeledFramesByFilter';
sourcePathToTakeTheFrames = './framesLabeled'
filterNames = ["canny","gaussian","median","cannyBlur","gamma"]
filterFrameIndex = 0;

os.makedirs(pathToMoveTheFrames, exist_ok=True);

foldersInSourcePath = os.listdir(sourcePathToTakeTheFrames);

for filterName in filterNames:
    print(f"COPYING {filterName} FILES")
    filterFrameIndex = 0
    pathToSave = f"{pathToMoveTheFrames}/{filterName}"
    os.makedirs(pathToSave, exist_ok=True)
    os.makedirs(f"{pathToSave}/yes", exist_ok=True)
    os.makedirs(f"{pathToSave}/no", exist_ok=True)
    for folder in foldersInSourcePath:
        print(f"=== coping {folder} frames...")
        patternFileName = f"{sourcePathToTakeTheFrames}/{folder}/*_{filterName}_*.png";
        framesWithTheFilter = glob.glob(patternFileName);
        for frameName in framesWithTheFilter:
            labelFirstLetter = frameName[frameName.rfind('/')+1];
            if labelFirstLetter == "y":
                shutil.copy(frameName, f"{pathToSave}/yes/{filterFrameIndex}.png")
            else:
                shutil.copy(frameName, f"{pathToSave}/no/{filterFrameIndex}.png")
            filterFrameIndex += 1
    print("FINISH__")
                


