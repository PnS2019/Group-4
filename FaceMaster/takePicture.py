from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
from face_detect import get_face
from prediction import predict
from tensorflow.keras.models import load_model

model = load_model('/home/pi/Desktop/Group4/Group-4-master/FaceMaster/model/facial_expression_detection2.hdf5')
camera = PiCamera()
camera.resolution = (256,256)
camera.framerate = 120
rawCapture = PiRGBArray(camera, size=(256,256))

i = 0;

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
	image = frame.array
	cv2.imshow('frame', image)
	rawCapture.truncate(0)
	
	if cv2.waitKey(1) & 0xFF == ord('s'):
		cv2.imwrite('/home/pi/Desktop/FaceMasters/PiCameraPictures/Original/{}.png'.format(i), image)
		print('Picture taken')
		
		try:
			image = get_face(image)
			cv2.imwrite('/home/pi/Desktop/FaceMasters/PiCameraPictures/Face/{}.png'.format(i), image)
			print('Face detected')
			image = image.reshape((1, 48, 48, 1))/255.
			predict(image, model)
		except IndexError:
			print('No face detected, please try again')
			
		i = i+1
		
		
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	#	break
