import os
import ijson
import cv2
import math

outputPath = "./jLeagueMatches/3925410/eventsFrames"
matchEventsJsonFilePath = "./jLeagueMatches/3925410/match_3925410_events.json"
firstHalfMatchVideoPath = "./jLeagueMatches/3925410/firstHalf.mp4"
secondHalfMatchVideoPath = "./jLeagueMatches/3925410/secondHalf.mp4"

os.makedirs(f"{outputPath}/pass", exist_ok=True);
os.makedirs(f"{outputPath}/no_pass", exist_ok=True);

firstHalfVideo = cv2.VideoCapture(firstHalfMatchVideoPath);
firstHalfVideoFPS = firstHalfVideo.get(cv2.CAP_PROP_FPS);
secondHalfVideo = cv2.VideoCapture(secondHalfMatchVideoPath);
secondHalfVideoFPS = secondHalfVideo.get(cv2.CAP_PROP_FPS);

initialFrameIndex = 0;
endFrameIndex = 0;
outputPathToSaveTheFrames = "";

""" with open(matchEventsJsonFilePath, "r", encoding="utf-8") as jsonFile:

    events = list(ijson.items(jsonFile, "item"))

    for idx, event in enumerate(events):
        listOfPassesSaved = os.listdir(f"{outputPath}/pass")
        listOfNotPassesSaved = os.listdir(f"{outputPath}/no_pass")
        eventName = event.get("id")
        print(f"Extracting event {eventName}...")
        if eventName in listOfPassesSaved or eventName in listOfNotPassesSaved:
            print("--- Event already saved")
            pass
        if event.get("type.name") == "Ball Receipt*":
            print("--- It's a ball receipt, going to search for the pass.")

            relatedEventId = event.get("related_events")
            passEventRelated = ""
            indexToSearch = idx - 1
            
            while events[indexToSearch].get("id") != relatedEventId and indexToSearch >= 0:
                indexToSearch -= 1
            if indexToSearch >= 0 and events[indexToSearch].get("type.name") == "Pass":
                passEventRelated = events[indexToSearch]
            else:
                pass
                
            if passEventRelated == "":
                print("--- Pass event not found.")
                pass

            if passEventRelated != "" and passEventRelated.get("period") == 1 and event.get("period") == 1:
                initialTime = passEventRelated.get("minute") * 60 + passEventRelated.get("second")
                endTime =  event.get("minute")  * 60 + event.get("second") + math.ceil(event.get("duration", 1.0))
                initialFrameIndex = int(initialTime * firstHalfVideoFPS)
                endFrameIndex = int(endTime * firstHalfVideoFPS)
                
                total_frames = int(firstHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

                if endFrameIndex - initialFrameIndex != 128:
                    endFrameIndex = min(initialFrameIndex + 128, total_frames)

                firstHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
                print("--- It's a good pass going to save.")
                os.makedirs(f"{outputPath}/pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/pass/{eventName}"
                eventFrameIndex = 0
                for frame in range(initialFrameIndex, endFrameIndex ):
                    ret, frame = firstHalfVideo.read()
                    if not ret:
                        break
                        
                    cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                    eventFrameIndex += 1
            else:
                initialTime = (event.get("minute")-45) * 60 + event.get("second")
                endTime =  (event.get("minute")-45) * 60 + event.get("second") + math.ceil(event.get("duration", 1.0))
                initialFrameIndex = int(initialTime * secondHalfVideoFPS)
                endFrameIndex = int(endTime * secondHalfVideoFPS)

                total_frames = int(secondHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

                if endFrameIndex - initialFrameIndex != 128:
                    endFrameIndex = min(initialFrameIndex + 128, total_frames)

                secondHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
                print("--- It's a good pass going to save.")
                os.makedirs(f"{outputPath}/pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/pass/{eventName}"
                eventFrameIndex = 0
                for frame in range(initialFrameIndex, endFrameIndex ):
                    ret, frame = secondHalfVideo.read()
                    if not ret:
                        break
                        
                    cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                    eventFrameIndex += 1
        elif event.get("type.name") != "Ball Receipt*" and event.get("type.name") != "Pass":
            if event.get("period") == 1:
                initialTime = event.get("minute") * 60 + event.get("second")
                durantion = math.ceil(event.get("duration", 2.0))
                endTime = initialTime + durantion
                initialFrameIndex = int(initialTime * firstHalfVideoFPS)
                endFrameIndex = int(endTime * firstHalfVideoFPS)
                total_frames = int(firstHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

                if endFrameIndex - initialFrameIndex != 128:
                    endFrameIndex = min(initialFrameIndex + 128, total_frames)
                firstHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
                print("--- It's not a pass going to save.")
                os.makedirs(f"{outputPath}/no_pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/no_pass/{eventName}"
                eventFrameIndex = 0
                for frame in range(initialFrameIndex, endFrameIndex):
                    ret, frame = firstHalfVideo.read()
                    if not ret:
                        break
                        
                    cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                    eventFrameIndex += 1
            else:
                initialTime = (event.get("minute")-45) * 60 + event.get("second")
                durantion = math.ceil(event.get("duration", 2.0))
                endTime = initialTime + durantion
                initialFrameIndex = int(initialTime * secondHalfVideoFPS)
                endFrameIndex = int(endTime * secondHalfVideoFPS)
                total_frames = int(secondHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

                if endFrameIndex - initialFrameIndex != 128:
                    endFrameIndex = min(initialFrameIndex + 128, total_frames)
                secondHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
                print("--- It's not a pass going to save.")
                os.makedirs(f"{outputPath}/no_pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/no_pass/{eventName}"
                eventFrameIndex = 0
                for frame in range(initialFrameIndex, endFrameIndex ):
                    ret, frame = secondHalfVideo.read()
                    if not ret:
                        break
                        
                    cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                    eventFrameIndex += 1 """

with open(matchEventsJsonFilePath, "r", encoding="utf-8") as jsonFile:
    events = ijson.items(jsonFile, "item")

    for event in events:
        eventName = event.get("id")
        print(f"-- checking event {eventName}")
        if event.get("period") == 1:
            print("-- it an event from period 1")
            initialTime = event.get("minute") * 60 + event.get("second")
            durantion = math.ceil(event.get("durations", 2.0)) + 1
            endTime = initialTime + durantion
            initialFrameIndex = int(initialTime * firstHalfVideoFPS)
            endFrameIndex = int(endTime * firstHalfVideoFPS)

            total_frames = int(firstHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

            if endFrameIndex - initialFrameIndex != 128:
                endFrameIndex = min(initialFrameIndex + 128, total_frames)

            firstHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
            if event.get("type.name") == "Pass":
                print("-- its a pass going to save it")
                os.makedirs(f"{outputPath}/pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/pass/{eventName}"
            elif event.get("type.name") != "Ball Receipt*":
                print("-- its not a pass going to save it")
                
                os.makedirs(f"{outputPath}/no_pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/no_pass/{eventName}"
            
            print(f"-- inital frame {initialFrameIndex} end {endFrameIndex}")
            
            eventFrameIndex = 0
            for frame in range(initialFrameIndex, endFrameIndex):
                ret, frame = firstHalfVideo.read()
                if not ret:
                    break
                    
                cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                eventFrameIndex += 1
        else:
            print("-- it an event from period 2")

            initialTime = (event.get("minute") - 45) * 60 + event.get("second")
            durantion = math.ceil(event.get("durations", 2.0)) + 1
            endTime = initialTime + durantion
            initialFrameIndex = int(initialTime * secondHalfVideoFPS)
            endFrameIndex = int(endTime * secondHalfVideoFPS)

            total_frames = int(secondHalfVideo.get(cv2.CAP_PROP_FRAME_COUNT))

            if endFrameIndex - initialFrameIndex != 128:
                endFrameIndex = min(initialFrameIndex + 128, total_frames)

            secondHalfVideo.set(cv2.CAP_PROP_POS_FRAMES, initialFrameIndex)
            if event.get("type.name") == "Pass":
                print("-- its a pass going to save it")
                os.makedirs(f"{outputPath}/pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/pass/{eventName}"
            elif event.get("type.name") != "Ball Receipt*":
                print("-- its not a pass going to save it")
                os.makedirs(f"{outputPath}/no_pass/{eventName}", exist_ok=True)
                outputPathToSaveTheFrames = f"{outputPath}/no_pass/{eventName}"
            eventFrameIndex = 0

            print(f"-- inital frame {initialFrameIndex} end {endFrameIndex}")


            for frame in range(initialFrameIndex, endFrameIndex):
                ret, frame = secondHalfVideo.read()
                if not ret:
                    break
                    
                cv2.imwrite(f"{outputPathToSaveTheFrames}/{eventFrameIndex}.jpg", frame)
                eventFrameIndex += 1