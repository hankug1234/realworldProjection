import numpy as np

def reOdering(image,size):
    w_state = image[0][size - 1][0][0] - image[0][0][0][0]
    h_state = image[0][-1][0][1] - image[0][0][0][1]

    # true right or up or horizontal  false left or down or vertical

    if(w_state>0):
        right = False
    else:
        right = True

    if(h_state>0):
        up = False
    else:
        up = True

    w_state = image[0][1][0][0] - image[0][0][0][0]
    h_state = image[0][1][0][1] - image[0][0][0][1]

    if(w_state>h_state):
        horizontal = True
    else:
        horizontal = False


    if(right==False and up == False and horizontal == False):
     objp = np.zeros((size * size, 3), np.float32)
     objp[:, :2] = np.mgrid[0:size, 0:size].T.reshape(-1, 2)

    elif(right==False and up == False and horizontal == True):
     objp = np.zeros((size * size, 3), np.float32)
     x,y = np.mgrid[0:size,0:size]
     n = np.array([y,x])
     n = n.T.reshape(-1,2)
     objp[:,:2] = n
     objp = [objp]

    elif(right==False and up == True and horizontal == False):
     objp = np.zeros((size*size,3),np.float32)
     n = np.mgrid[0:size,0:size].T
     for s in range(0,len(n)):
      n[s] = np.sort(n[s],axis=0)[::-1]
     objp[:,:2] = n.reshape(-1,2)
     objp = [objp]

    elif(right==False and up == True and horizontal == True):
     objp = np.zeros((size * size, 3), np.float32)
     x,y = np.mgrid[0:size,0:size]
     n = np.array([y,x])
     n = n.T
     n = n[::-1]
     objp[:,:2] = n.reshape(-1,2)
     objp = [objp]

    elif(right==True and up == True and horizontal == True):
     objp = np.zeros((size * size, 3), np.float32)
     x,y = np.mgrid[0:size,0:size]
     n = np.array([y,x])
     n = n.T
     n = n[::-1]
     for s in range(0,len(n)):
      n[s] = np.sort(n[s],axis=0)[::-1]
     objp[:,:2] = n.reshape(-1,2)
     objp = [objp]


    elif(right==True and up == True and horizontal == False):
     objp = np.zeros((size * size, 3), np.float32)
     n = np.mgrid[0:size,0:size]
     n = n.T
     n = n[::-1]
     for s in range(0,len(n)):
      n[s] = np.sort(n[s],axis=0)[::-1]
     objp[:,:2] = n.reshape(-1,2)
     objp = [objp]

    elif(right==True and up == False and horizontal == False):
     objp = np.zeros((size * size, 3), np.float32)
     n = np.mgrid[0:size,0:size]
     n = n.T
     n = n[::-1]
     objp[:,:2] = n.reshape(-1,2)
     objp = [objp]

    else:
      objp = np.zeros((size * size, 3), np.float32)
      x,y = np.mgrid[0:size,0:size]
      n = np.array([y,x])
      n = n.T
      for s in range(0,len(n)):
       n[s] = np.sort(n[s],axis=0)[::-1]
      objp[:,:2] = n.reshape(-1,2)
      objp = [objp]

    return objp