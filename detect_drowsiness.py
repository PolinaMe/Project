# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import math
import matplotlib.pyplot as plt

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def smile_aspect_ratio(mouth):
	A = dist.euclidean(mouth[0], mouth[9])
	B = dist.euclidean(mouth[6], mouth[9])
	C = dist.euclidean(mouth[0], mouth[6])
	L=(math.pow(A,2.0)+math.pow(B,2.0)-math.pow(C,2.0))/(2.0*A*B)
	
	# return the mouth aspect ratio
	return math.degrees(math.acos(L))

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="C:/Users/Polina/project/alarm.wav",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=1,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3 # blink indicator 
EYE_AR_CONSEC_FRAMES = 40 # consecutive frames for closesd eye 
arr_EAR=[] # used for fatigue chart
arr_time=[] # used for fatigue chart

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0 # count frames with closed eyes
ALARM_ON = False

# initialize general variables for further calculations
frame_counter=0 # general frame counter
minEAR=0.5 # initialized min eye ratio for further calculations
maxEAR=0 # initialized max eye ratio for further calculations
countEAR=0 #EAR average counter
countMAR=0 #MAR average counter
STD=0 # standard deviation

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# grab the indexes of the facial landmarks for the mouth 
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
writer=None

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize it, 
	# and convert it to grayscale channels
	frame = vs.read()
	# if the frame was not grabbed, then we have reached the end of the stream
	# if not grabbed: break
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		# same for mouth
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth= shape[mStart:mEnd]

		# calculating ratios respectively
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mar= smile_aspect_ratio(mouth)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		# same for mouth
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)

		frame_counter+=1
		countMAR+=mar
		countEAR+=ear

		# for the first 300 frames, calculate the average shapes of the face: eyes and mouth
		if frame_counter<300:
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
			cv2.putText(frame, "FACE RECOGNITION: {:.0f}%".format(frame_counter/3), (150, 250),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			# if the average is lower than the minimum eyes ratio - update min eye rat
			if ear<minEAR:
				minEAR=ear
			# if the average is higher than the maximum eyes ratio - update max eye rat
			if ear>maxEAR:
				maxEAR=ear
		# when frame counter reach 300, make sure that the eye ratio is within the range of 
		# eyes receptivity of the same face
		elif frame_counter==300:
			if (maxEAR+minEAR)/2>0.15 and (maxEAR+minEAR)/2<0.35:
				EYE_AR_THRESH=(maxEAR+minEAR)/2 # updating the blink/closed eye threshold
			print (EYE_AR_THRESH)

		# every 100 frames, calculating average of ashape and adding it to chart
		else:
			if frame_counter%100==0:
				arr_EAR.append(countEAR/100)
				arr_time.append(frame_counter/100)
				print("~[frame #%d] MAR: %f " % (frame_counter,countMAR/100)) 
				countEAR=0
				countMAR=0

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH and mar>135:#and dist.euclidean(mouth[14],mouth[18])<9):
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
				cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
				COUNTER += 1
				STD=0
			# if the eyes were closed for a sufficient number of frames
			# then sound the alarm
				if COUNTER >= EYE_AR_CONSEC_FRAMES :
				# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
						if args["alarm"] != "":
							t = Thread(target=sound_alarm, args=(args["alarm"],))
							t.deamon = True
							t.start()
					# if there is no more need for alarm, then sound goes off 
					if not t.isAlive():
						ALARM_ON=False
					# wake up drawn on the frame
					cv2.putText(frame, "wake up!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
			else:
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
				STD+=1
				if STD>5:
					COUNTER = 0
					ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counterscd 
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)
			cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)
			cv2.putText(frame, "Frame: {:}".format(frame_counter), (125, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)

			# writing video to hard system
			if writer is None :
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter("output.avi", fourcc, 20,
					(frame.shape[1], frame.shape[0]), True)
			if writer is not None:
				writer.write(frame)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# showing plot at the end for running
plt.plot(arr_time,arr_EAR)
plt.ylim(0.2,0.4)
plt.show()
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()