from detectPlate import detect
from imutils.video import VideoStream
import argparse
import time
import cv2

# Set webcam as video source
video_source = VideoStream(src=0).start()
time.sleep(2.0)

while True:

	# If there's no video source, break loop
	if video_source is None: 
		break

	# Read one frame
	frame = video_source.read()
	if frame is None:
		break
	
	# Detect plates
	(box, plate) = detect(frame)

	if box is not None:
		# Draw contour
		cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)

		# Display license plate on screen
		cv2.putText(frame, 
                'Patente: ' + plate, 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

	# Show frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# End program if user presses 'q' on their keyboard
	if key == ord("q"):
		break

# Release video source and close window 
video_source.stop()
cv2.destroyAllWindows()