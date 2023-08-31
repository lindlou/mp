# https://stackoverflow.com/questions/37774354/opencv-python-real-time-image-frame-processing

import numpy as np
import cv2
import time

#res = 32
#camera = picamera.PiCamera()
#camera.resolution = (res,res)

def do_a_count(delay):
    print("capture1")
    ret, frame1 = cap.read()
    time.sleep(delay)
    print("capture2")
    ret, frame2 = cap.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.GaussianBlur(frame1, (21, 21), 0)

    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.GaussianBlur(frame2, (21, 21), 0)

    frameDelta = cv2.absdiff(frame1, frame2)
    # frameDelta = cv2.imread("cells.png")
    # frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
    # frameDelta = cv2.GaussianBlur(frameDelta, (21, 21), 0)

    # https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frameDelta)

    print(len(keypoints))
    
    #return(len(keypoints))
    frameDeltaKeypoints = cv2.drawKeypoints(frameDelta, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", frameDeltaKeypoints)
    k = cv2.waitKey(0)


cap = cv2.VideoCapture(0)

do_a_count(1)
print("done")
#time.sleep(3)

cap.release()
cv2.destroyAllWindows()

