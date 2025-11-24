import cv2
import numpy as np

gray_images = []
objpoints = []
imgpoints = []
pattern_size = (6, 4)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

for i in range(1, 11):
	gray = cv2.cvtColor(cv2.imread("chessboard images/%s.jpg" % i), cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=None)
	objpoints.append(objp)
	imgpoints.append(corners)
	gray_images.append(gray)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
	objpoints, imgpoints, gray_images[0].shape[::-1], None, None
)
print(mtx)