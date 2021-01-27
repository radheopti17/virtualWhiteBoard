import numpy as np
import cv2
from collections import deque
from datetime import datetime
import os

directory= r'/media/rushikesh/F/Painting-With-Finger-Gestures-master/savedimg'
flag=1
# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]


bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Setup the Paint interface
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.circle(paintWindow, (120,400), 30, (0,0,0), 2)
paintWindow = cv2.circle(paintWindow, (200,400),30, colors[0], -1)
paintWindow = cv2.circle(paintWindow, (280,400), 30, colors[1], -1)
paintWindow = cv2.circle(paintWindow, (360, 400),30, colors[2], -1)
paintWindow = cv2.circle(paintWindow, (440,400), 30,colors[3], -1)
paintWindow = cv2.circle(paintWindow, (520,400), 30, (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (590,10), (620,40), colors[2], -1)
cv2.putText(paintWindow, "CLEAR", (100, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (183, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (260, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (340, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (420, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "SAVE", (500, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "X", (595, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Load the video
camera = cv2.VideoCapture(0)

# Keep looping
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Add the coloring options to the frame
    frame = cv2.circle(frame, (120,400), 30, (122,122,122), -1)
    frame = cv2.circle(frame, (200,400),30, colors[0], -1)
    frame = cv2.circle(frame, (280,400), 30, colors[1], -1)
    frame = cv2.circle(frame, (360, 400),30, colors[2], -1)
    frame = cv2.circle(frame, (440,400), 30, colors[3], -1)
    frame = cv2.circle(frame, (520,400), 30, (122,122,122), -1)
    frame = cv2.rectangle(frame, (590,10), (620,40), colors[2], -1)
    cv2.putText(frame, "CLEAR", (100, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (183, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (260, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "RED", (340, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (420, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(frame, "SAVE", (500, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "X", (595, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Check to see if we have reached the end of the video
    if not grabbed:
        break

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours in the image
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
    	# Sort the contours and find the largest one -- we
    	# will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1]>=360:
            if 90 <= center[0] <= 150: # Clear All
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]


                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0


                paintWindow[67:,:,:] = 255

            elif 170 <= center[0] <= 230:
                    colorIndex = 0 # Blue
            elif 250 <= center[0] <= 310:
                    colorIndex = 1 # Green
            elif 330 <= center[0] <= 390:
                    colorIndex = 2 # Red
            elif 410 <= center[0] <= 470:
                    colorIndex = 3 # Yellow
            elif (490 <= center[0] <= 550) and flag:

                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    filename='imgat '+str(current_time)+'.jpg'
                    os.chdir(directory)
                    cv2.imwrite(filename, paintWindow)
                    flag=0

        elif(center[1]>=10 and center[1]<=40 and center[0]<=620 and center[0]>=590):
            break
        else :
            flag=1
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)
            elif colorIndex == 3:
                ypoints[yindex].appendleft(center)

    # Append the next deque when no contours are detected (i.e., bottle cap reversed)
    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        ypoints.append(deque(maxlen=512))
        yindex += 1

    # Draw lines of all the colors (Blue, Green, Red and Yellow)
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

	# If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
