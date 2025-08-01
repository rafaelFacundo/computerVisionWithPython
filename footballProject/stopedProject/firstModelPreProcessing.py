import os
import cv2


videosFolderPath = './videos';

folderFiles = os.listdir(videosFolderPath)

i = 0

video = cv2.VideoCapture("./videos/match1.mp4")

while video.isOpened():
    ret, frame = video.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()

""" for file in folderFiles:
    if ".mp4" in file:
        print(file.)
        video = cv2.VideoCapture(f"./videos/{file}")
        while video.isOpened():
            ret, frame = video.read();
            if not ret:
                print("can't resolve frame")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame", gray)
            if cv2.waitKey(1) == ord('q'):
                break
            
        video.release()
        cv2.destroyAllWindows() """