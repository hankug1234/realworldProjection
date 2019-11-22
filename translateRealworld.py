import numpy as np
import math


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
