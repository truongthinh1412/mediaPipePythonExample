import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, mode=False, maxHand=2, detectionThreshold=0.5, trackThreshold=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionThreshold = detectionThreshold
        self.trackThreshold = trackThreshold

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHand, detectionThreshold, trackThreshold)

        # for drawing
        self.mpDraws = mp.solutions.drawing_utils

    def findLandmark(self, img, handNo):
        lmList = []
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            for idLm, lms in enumerate(myHand.landmark):
                cx, cy = int(lms.x * w), int(lms.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def findHand(self, video, draw=True):
        cap = cv2.VideoCapture(video)
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, img = cap.read()

            if ret:
                # img size
                h, w, c = img.shape
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)
                # loop through each hand
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        if draw:
                            self.mpDraws.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                        # landmark is based on image size, multiply landmark x,y,z by image size
                        # to get location of landmark on image(x multiply with width, y multiply with height)
                        for idLm, lm in enumerate(handLms.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            print(idLm, cx, cy)
                            if draw:
                                cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

                cv2.imshow('Frame', img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()


def main():
    cTime = 0
    pTime = 0

    handDetector = HandDetector()
    # handDetector.findHand("sampleHand.mp4", draw=False)
    cap = cv2.VideoCapture("sampleHand.mp4")
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()

        if ret:
            listLM = handDetector.findLandmark(img, 0)
            for landmark in listLM:
                cv2.circle(img, (landmark[1], landmark[2]), 10, (255, 0, 0), cv2.FILLED)
            cv2.imshow('Frame', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
