import os
import cv2
import glob

def labelFrameYesOrNo(videoPath):
    video = cv2.VideoCapture(videoPath);
    pathToSaveFrames = f"{videoPath}/frames"
    os.makedirs(pathToSaveFrames, exist_ok=True)
    index = 0;
    while video.isOpened():
        print(f"frame index is {index}")
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read();
        if not ret:
            print("can't resolve frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)
        key = cv2.waitKey(0) & 0xFF
        grayImageResized = cv2.resize(
            gray, 
            (112, 112),
            interpolation=cv2.INTER_LINEAR
        )
        if key == ord('b'):
            if index > 0:
                index -= 1
                print(f"rollback going to delete {index}")
                patternFileName = os.path.join(pathToSaveFrames, f"*_{index}.png")
                filesFound = glob.glob(patternFileName);
                for file in filesFound:
                    os.remove(file)
            continue
        elif key == ord('y'):
            cv2.imwrite(os.path.join(pathToSaveFrames, f"yes_{index}.png"), grayImageResized)
            index += 1
        elif key == ord('n'):
            cv2.imwrite(os.path.join(pathToSaveFrames, f"no_{index}.png"), grayImageResized)
            index += 1
        elif key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()