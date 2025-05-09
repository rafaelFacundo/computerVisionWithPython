import os
import cv2
import glob


def labelFrameYesOrNo(videoPath, resutlFolderName, filters):
    index = 0;
    video = cv2.VideoCapture(videoPath);
    pathToSaveFrames = resutlFolderName
    os.makedirs(pathToSaveFrames, exist_ok=True)

    if os.path.isfile(f"{resutlFolderName}/lastIndex.txt"):
        indexFile = open(f"{resutlFolderName}/lastIndex.txt", "r")
        fileContent = indexFile.read().strip()
        if fileContent != "":
            index = int(fileContent)
    else:
        indexFile = open(f"{resutlFolderName}/lastIndex.txt", "r")
        indexFile.write(str(0))
        indexFile.flush();
    
    indexFile.close();

    while video.isOpened():
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read();
        if not ret:
            print("can't resolve frame")
            break
        cv2.imshow("frame", frame)
        key = cv2.waitKey(0) & 0xFF
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayImageResized = cv2.resize(
            gray, 
            (320, 320),
            interpolation=cv2.INTER_LINEAR
        )
        if key == ord('b'):
            if index > 0:
                index -= 1
                patternFileName = os.path.join(pathToSaveFrames, f"*_{index}.png")
                filesFound = glob.glob(patternFileName);
                for file in filesFound:
                    os.remove(file)
                indexFile = open(f"{resutlFolderName}/lastIndex.txt", "w+")
                indexFile.write(str(index))
                indexFile.flush()
                indexFile.close()
            continue
        elif key == ord('y'):
            for filterObject in filters:
                cv2.imwrite(
                    os.path.join(
                        pathToSaveFrames,
                        f"yes_{filterObject["name"]}_{index}.png"
                    ),
                    filterObject["filter"](grayImageResized)
                )
            index += 1
            indexFile = open(f"{resutlFolderName}/lastIndex.txt", "w+")
            indexFile.write(str(index))
            indexFile.flush()
            indexFile.close()
        elif key == ord('n'):
            for filterObject in filters:
                cv2.imwrite(
                    os.path.join(
                        pathToSaveFrames,
                        f"no_{filterObject["name"]}_{index}.png"
                    ),
                    filterObject["filter"](grayImageResized)
                )
            index += 1
            indexFile = open(f"{resutlFolderName}/lastIndex.txt", "w+")
            indexFile.write(str(index))
            indexFile.flush()
            indexFile.close()
        elif key == ord('q'):
            break
    
   
    video.release()
    cv2.destroyAllWindows()