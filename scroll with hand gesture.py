import os
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui  # Library to control the mouse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb and fingers' tip landmarks
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)

        return self.lmList, bbox

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        if len(self.lmList) < max(p1, p2) + 1:
            return 0, img, []

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = handDetector()
    screen_width, screen_height = pyautogui.size()
    cam_width, cam_height = 640, 480
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    prev_wrist_y = None  # Variable to track wrist position for scrolling

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            # Get the wrist and finger tip positions
            wrist_x, wrist_y = lmList[0][1], lmList[0][2]  # Wrist position (id 0)
            index_x, index_y = lmList[8][1], lmList[8][2]  # Index finger tip (id 8)

            # Convert index finger tip to screen coordinates
            screen_x = np.interp(index_x, (0, cam_width), (0, screen_width))
            screen_y = np.interp(index_y, (0, cam_height), (0, screen_height))

            # Move the mouse to the screen coordinates
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate the distance between index and thumb for left click
            length_index_thumb, img, _ = detector.findDistance(8, 4, img)

            if length_index_thumb < 30:
                pyautogui.click()
                cv2.putText(img, "Left Click", (index_x, index_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Scrolling Gesture (using wrist movement)
            if prev_wrist_y is not None:
                if wrist_y < prev_wrist_y:  # Hand moving up
                    pyautogui.scroll(50)  # Scroll up
                    cv2.putText(img, "Scroll Up", (wrist_x, wrist_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                elif wrist_y > prev_wrist_y:  # Hand moving down
                    pyautogui.scroll(-50)  # Scroll down
                    cv2.putText(img, "Scroll Down", (wrist_x, wrist_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # Update the wrist position for the next frame
            prev_wrist_y = wrist_y

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        # Exit loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
