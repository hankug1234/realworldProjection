import numpy as np
import math
import cv2
import numpy.linalg as lin
import glob

def homogeneousCoordinate(vec):
    return np.array([vec[0],vec[1],1])
def deHomogeneousCoordinate(vec):
    return np.array([vec[0],vec[1]])

def toNormalPlane(inv, vec):
    return np.dot(inv,vec)

def distort_normal(vec, distortV):
      dist = distortV[0]
      r2 = vec[0]**2 + vec[1]**2;
      radial_d = 1 + dist[0] * r2 + dist[1] * r2**2 + dist[4] * r2**3;
      x_d = radial_d * vec[0] + 2 * dist[2] * vec[0] * vec[1] + dist[3] * (r2 + 2 * vec[0]**2);
      y_d = radial_d * vec[1] + dist[2] * (r2 + 2 * vec[1]**2) + 2 * dist[3] * vec[0] * vec[1];
      return np.array([x_d,y_d])


def undistort(vec, distortV,thresh):
  p_d = vec
  p_u = vec
  while(1):
   dist = distort_normal(p_u,distortV)
   err = (dist - p_d)
   p_u = (p_u - err)
   if((math.fabs(err[0])<thresh) and (math.fabs(err[1])<thresh)):
     break
  return p_u

def toRealworld(inv,vec,transv):
    zero = np.array([0,0,0])
    transv = np.array(transv,dtype='f')
    transv = transv.reshape(-1,1).transpose()
    tZero = zero-transv[0]
    tVec = vec-transv[0]
    Rzero = np.dot(inv,tZero)
    Rvec = np.dot(inv,tVec)

    dr = Rvec - Rzero
    k = -(Rzero[2]/dr[2])
    re =[Rzero[0]+k*dr[0],Rzero[1]+k*dr[1],Rzero[2]+k*dr[2]]
    return re

def translateToRealworld(vec,inMtx,inRevecsMetrix,tvecs,dist,thresh):
    vec = homogeneousCoordinate(vec)
    vec = toNormalPlane(inMtx,vec)
    vec = deHomogeneousCoordinate(vec)
    vec = undistort(vec,dist,thresh)
    vec = homogeneousCoordinate(vec)
    vec = toRealworld(inRevecsMetrix,vec,tvecs)
    return vec

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

print(translateToRealworld(a,inMtx,inRvecsMetrix,tvecs,dist,0.0001))
print(b)


cv2.destroyAllWindows()



