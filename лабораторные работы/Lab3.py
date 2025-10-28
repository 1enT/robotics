import cv2
import time

fps_arr = []
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')
while True:
	start_time = time.time()
	ret, frame = cap.read()
	blured_frame = cv2.GaussianBlur(frame, (3, 3), 0)
	gray_frame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2GRAY)
	eyes, smiles = [], []

	faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=10, minSize=(30, 30), maxSize=(500, 500))
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roi_gray = gray_frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20, minSize=(15, 15), maxSize=(100, 100))
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(frame, (x+ex, y+ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
		smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.6, minNeighbors=50, minSize=(40, 40), maxSize=(400, 400))
		for (ex, ey, ew, eh) in smiles:
			cv2.rectangle(frame, (x+ex, y+ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

	time_diff = time.time() - start_time
	fps = round(1/time_diff*10)/10
	fps_arr.append(fps)
	cv2.putText(frame, "FPS: " + str(fps), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
	if len(eyes) < 2:
		cv2.putText(frame, "Open your eyes!", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
	if len(smiles) == 0:
		cv2.putText(frame, "Smile!", (0, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow('', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
print("Average FPS {}".format(sum(fps_arr)/len(fps_arr)))