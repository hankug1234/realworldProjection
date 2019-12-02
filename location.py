import numpy as np
import cv2
import numpy.linalg as lin
import ordering
import translate_to_real_world

# termination criteria
rectangleSize = (31.5,31.5)#size cm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((7 * 7, 3), np.float32)

objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.

objpoints = []  # 3d point in real world space

imgpoints = []  # 2d points in image plane.

# cv2.namedWindow('KnV')

images = ['C:/Users/LG/Pictures/Camera Roll/board3.jpg']

for fname in images:
    img = cv2.imread(fname)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners

    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        #cv2.imshow('KnV', img)

        #cv2.waitKey(1000)

        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners)

        # Draw and display the corners

        cv2.drawChessboardCorners(gray, (7, 7), corners, ret)

        cv2.imshow('KnV', gray)

        cv2.waitKey()


objpoints= ordering.reOdering(imgpoints,7)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
rvecsMatrix, J = cv2.Rodrigues(rvecs[0])
h, w = img.shape[:2];
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

inMtx = lin.inv(mtx)
inRvecsMetrix = lin.inv(rvecsMatrix)
inNewMtx = lin.inv(newcameramtx)

a = imgpoints[0][10][0]
b = objpoints[0][10]

print(objpoints)
print("")
print(imgpoints)

print(translateRealworld.translateToRealworld(a,inMtx,inRvecsMetrix,tvecs,dist,0.0001))
print(b)


cv2.destroyAllWindows()



