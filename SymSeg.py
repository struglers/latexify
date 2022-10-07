# comments: in the thresh image: Black pixel has value 0, white pixel has value 255
# image has coordinates img[X][Y] which is the conventional form




from re import X
import cv2
import math
import numpy as np

#HyperParameters
num_dirs = 18


def symSeg(im_loc, printOP = False):
    img = cv2.pyrDown(cv2.imread(im_loc, cv2.IMREAD_UNCHANGED))
    orig_img = img.copy()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    im_data = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S, 8)
    (numLabels, labels, stats, centroids) = im_data
    
    if printOP == True:
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            (cX, cY) = centroids[i]
            
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            img = cv2.circle(img, (int(cX), int(cY)), 4, (0,0,255), -1)
            img = cv2.putText(img, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 0),thickness=3)
            print(i, "X: ", cX, "Y: ", cY)
        cv2.imshow("Input Image", orig_img)
        cv2.imshow("Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return (centroids, stats, thresh)
    










def LOSViewer(centroids, stats, img):
    dir = []
    for x in range(num_dirs):
        dir.append((math.cos(2*math.pi*x/num_dirs), math.sin(2*math.pi*x/num_dirs)))

    break_count = 0
    for i in range(len(centroids)):
        for j in range(len(dir)):
            for n in range(2000):
                pX = centroids[i][0] + dir[j][0]*n
                pY = centroids[i][1] + dir[j][1]*n
                if(pX > img.shape[1] or pX < 0 or pY < 0 or pY > img.shape[0]):
                    break_count += 1
                    print(break_count, "break")
                    break
                img = cv2.circle(img, (int(pX), int(pY)), 0, (0,0,255), -1)

    cv2.imshow('', img)
    cv2.waitKey(0)











def LOSGraphBuilder(centroids, stats, img):
    cpy = img.copy()
    img_w = stats[0][cv2.CC_STAT_WIDTH]
    img_h = stats[0][cv2.CC_STAT_HEIGHT]
    dir = []
    graph = {}
    for x in range(num_dirs):
        dir.append((math.cos(2*math.pi*x/num_dirs), math.sin(2*math.pi*x/num_dirs)))

    for i in range(len(centroids)):
        graph[i] = []
        for j in range(len(dir)):
            n = 1
            while True:
                # move from centroid in direction dir until you find a white pixel, or you overflow the image dimensions
                pX = int(centroids[i][0] + dir[j][0]*n)
                pY = int(centroids[i][1] + dir[j][1]*n)

                if i == 8:
                    print(j, n, pX, pY)
                # problem Identified: Since 8 and 4 are so close, for each direction 8 always hits 4 first before reaching any other node and so 8 is only
                # connected to 4. This may be either desirable or a problem. Need to discuss this.


                # overflowed image dimensions?
                if(pX >= img_w or pX < 0 or pY < 0 or pY >= img_h):
                    break

                # found white pixel? Remember we are using 'thresh' as our image here and it is inverted in color
                if img[pY][pX] > 100:
                    flag = False
                    for bb in range(1, len(centroids)):
                        # search for the bounding box in which this pixel lies
                        # bb is an index into the bounding boxes list
                        if bb == i:
                            continue
                        if inBoundingBox((pX, pY), stats[bb]):
                            flag = True
                            if bb in graph[i]:
                                continue
                            else:
                                graph[i].append(bb)
                    if flag:
                        break

                cpy = cv2.circle(cpy, (int(pX), int(pY)), 0, (128, 128, 128), -1)
                
                n += 1
                
    for i in range(11):
        (cX, cY) = centroids[i]
        cpy = cv2.putText(cpy, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255),thickness=3)
    cv2.imshow('', cpy)
    cv2.waitKey(0)

    return graph

def inBoundingBox(point, dim):
    xmin = dim[cv2.CC_STAT_LEFT]
    ymin = dim[cv2.CC_STAT_TOP]
    xmax = dim[cv2.CC_STAT_LEFT] + dim[cv2.CC_STAT_WIDTH]
    ymax = dim[cv2.CC_STAT_TOP] + dim[cv2.CC_STAT_HEIGHT]
    if point[0] >= xmin and point[0] <= xmax and point[1] >= ymin and point[1] <= ymax:
        return True
    return False



(centroids, stats,img) = symSeg('test.jpg')
#print(img[250][100])
graph = LOSGraphBuilder(centroids, stats, img)

print(graph)
for i in range(11):
    (cX, cY) = centroids[i]
    img = cv2.putText(img, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255),thickness=3)
cv2.imshow('', img)
cv2.waitKey(0)

#cv2.imshow("Input Image", orig_img)
#cv2.imshow("Bounding Boxes", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img = cv2.circle(img, (100,120), 5, (0,0,255), -1)
#cv2.imshow('', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(inBoundingBox((100, 120), stats[6]))

#for i in range(120, len(img)):
#    img[i] = np.ones_like(img[i])*128
#cv2.imshow('', img)
#cv2.waitKey(0)