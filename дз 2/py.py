import cv2
import numpy as np
import math


low_red = np.array([28, 50, 50])
high_red = np.array([70, 255, 255])
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
	hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

	red_mask = cv2.inRange(hsv_frame, low_red, high_red)
	blurred_frame = cv2.GaussianBlur(blurred_frame, (101, 101), 0)
	greened_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=red_mask)

	gray_frame = cv2.cvtColor(greened_frame, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
		area = cv2.contourArea(contour)
		if(area > 4000):
			contour = cv2.fitEllipse(contour)
			center_x = int(contour[0][0])
			center_y = int(contour[0][1])
			radius = contour[1][0]/2
			angle = contour[2]
			cv2.ellipse(frame, contour, (0, 255, 0), 3)
			cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
			first_point_x = int(center_x + math.cos(math.pi/180*angle)*radius)
			first_point_y = int(center_y + math.sin(math.pi/180*angle)*radius)
			second_point_x = int(center_x - math.cos(math.pi/180*angle)*radius)
			second_point_y = int(center_y - math.sin(math.pi/180*angle)*radius)
			cv2.line(frame, (first_point_x, first_point_y), (second_point_x, second_point_y), (0, 0, 255), 2)

			dy = abs(first_point_y-second_point_y)
			dx = abs(first_point_x-second_point_x)
			angle_hor = int(180/math.pi*(math.atan(dy/dx)))
			angle_vert = 90-angle_hor
			cv2.putText(frame, "Contour center: {}, {}".format(int(contour[0][0]), int(contour[0][1])), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(frame, "Angle with the horizontal: {}".format(angle_hor), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(frame, "Angle with the vertical: {}".format(angle_vert), (0, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
			

	cv2.imshow('1', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()