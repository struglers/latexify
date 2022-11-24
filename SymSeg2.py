# comments: in the thresh image: Black pixel has value 0, white pixel has value 255
# image has coordinates img[X][Y] which is the conventional form

# ToDo: code cleanup, function comments


import cv2
import numpy as np
from operator import itemgetter

_cX = 0
_cY = 1
_x = 2
_y = 3
_w = 4
_h = 5

#HyperParameters
num_dirs = 8
image = '12.jpg'
threshold = 600 # downscale the input image to have atmost <threshold> number of pixels in its larger dimension

def process(img):
    '''
    import the image and convert it to the required size determined by the hyperparameter threshold
    Also, return the downscaled and grayscaled image
    '''
    while img.shape[0] > threshold or img.shape[1] > threshold:
        img = cv2.pyrDown(img)
    #orig_img = img.copy()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''gray_img = 255*(gray_img < 80).astype(np.uint8)
    coords = cv2.findNonZero(gray_img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h,x:x+w]
    gray_img = gray_img[y:y+h,x:x+w]
    '''
    return img, gray_img


def symSeg(im_loc, printOP = False):
    '''
    
    '''
    img = cv2.imread(im_loc)
    img, gray_img = process(img)
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    im_data = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S, 4)
    (numLabels, labels, stats, centroids) = im_data

    # filtering noise points by checking area
    # Actually I cannot do this, otherwise we cannot identify 'therefore' symbol
    '''
    temp_s = []
    temp_c = []
    for i in range(len(stats)):
        if stats[i, cv2.CC_STAT_AREA] > 40:
            temp_s.append(stats[i])
            temp_c.append(centroids[i])
    stats = temp_s
    centroids = temp_c
    '''
    for i in range(len(stats)):
        print(stats[i])
    
    coordinates = []
    for i in range(len(stats)):
        #print(stats[i])
        x = stats[i][cv2.CC_STAT_LEFT]
        y = stats[i][cv2.CC_STAT_TOP]
        w = stats[i][cv2.CC_STAT_WIDTH]
        h = stats[i][cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]
        coordinates.append([cX, cY, x ,y , w, h])
    #sorted(coordinates, key=itemgetter(0,1))
    #sorted(coordinates, key=itemgetter(0,1))
    
    coordinates.pop(0)
    for i in range(len(coordinates)):
        print(coordinates[i])

    coordinates = sorted(coordinates, key = itemgetter(_cX, _cY))
    
    for i in range(len(coordinates)):
        print(coordinates[i])

    
    if printOP == True:
        for i in range(len(coordinates)):
            #x = stats[i, cv2.CC_STAT_LEFT]
            #y = stats[i, cv2.CC_STAT_TOP]
            #w = stats[i, cv2.CC_STAT_WIDTH]
            #h = stats[i, cv2.CC_STAT_HEIGHT]
            #(cX, cY) = centroids[i]
            x,y = coordinates[i][_x], coordinates[i][_y]
            w,h = coordinates[i][_w], coordinates[i][_h]
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
            cX,cY = coordinates[i][_cX], coordinates[i][_cY]
            img = cv2.circle(img, (int(cX), int(cY)), 4, (0,0,255), -1)
            img = cv2.putText(img, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)
            #print(i, "X: ", cX, "Y: ", cY)
        
        cv2.imshow("gray", thresh)
        cv2.waitKey(0)
        #cv2.imshow("Input Image", orig_img)
        cv2.imshow("Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.imwrite('thresh.jpg', thresh)
    cv2.imwrite('BoundingBoxes.jpg', img)

    return (coordinates, thresh)
    
def LOSViewer(coordinates, graph):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img, gray_img = process(img)
    for snode in graph:
        for enode in graph[snode]:
            (sX, sY) = coordinates[snode][_cX], coordinates[snode][_cY]
            (eX, eY) = coordinates[enode][_cX], coordinates[enode][_cY]
            cv2.line(img, (int(sX), int(sY)), (int(eX), int(eY)), color = (0,0,255), thickness = 1)

    cv2.imshow('graph', img)
    cv2.waitKey(0)
    cv2.imwrite('graph.jpg', img)


def LOSGraphBuilder(coordinates, img):
    cpy = img.copy()
    img_w = img.shape[1]
    img_h = img.shape[0]
    dir = list()
    graph = dict()
    for x in range(num_dirs):
        dir.append((np.cos(2*np.pi*x/num_dirs), np.sin(2*np.pi*x/num_dirs)))

    for i in range(len(coordinates)):
        graph[i] = []
        for j in range(len(dir)):
            n = 1
            while True:
                # move from centroid in direction dir until you find a white pixel, or you overflow the image dimensions
                pX = int(coordinates[i][_cX] + dir[j][0]*n)
                pY = int(coordinates[i][_cY] + dir[j][1]*n)

                #if i == 8:
                #    print(j, n, pX, pY)
                # problem Identified: Since 8 and 4 are so close, for each direction 8 always hits 4 first before reaching any other node and so 8 is only
                # connected to 4. This may be either desirable or a problem. Need to discuss this.


                # overflowed image dimensions?
                if(pX >= img_w or pX < 0 or pY < 0 or pY >= img_h):
                    break

                # found white pixel? Remember we are using 'thresh' as our image here and it is inverted in color
                if img[pY][pX] > 100:
                    # flag checks whether we have encountered a new symbol or we are still hitting
                    # the current symbol while looking in LOS.
                    flag = False
                    for bb in range(1, len(coordinates)):
                        # search for the bounding box in which this pixel lies
                        # bb is an index into the bounding boxes list
                        if bb == i:
                            continue
                        if inBoundingBox((pX, pY), coordinates[bb]):
                            flag = True
                            if bb in graph[i]:
                                continue
                            else:
                                graph[i].append(bb)
                    if flag:
                        break

                cpy = cv2.circle(cpy, (int(pX), int(pY)), 0, (128, 128, 128), -1)
                
                n += 1
    
    # part of code which makes this an undirected graph
    for i in graph.keys():
        for j in graph[i]:
            if i not in graph[j]:
                graph[j].append(i)

    # Convert graph to a representation which can be fed into the GNN
    data = [[],[]]   # this list will be converted to a numpy array later
    for i in graph.keys():
        for j in graph[i]:
            data[0].append(i)
            data[1].append(j)

    data = np.array(data)

    for i in range(len(coordinates)):
        (cX, cY) = coordinates[i][_cX], coordinates[i][_cY]
        cpy = cv2.putText(cpy, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
    cv2.imshow('LOS', cpy)
    cv2.waitKey(0)
    cv2.imwrite('LOS.jpg', cpy)

    return graph, data
    # graph is adjacency list representation, data is a specific representation fed to GNN

def inBoundingBox(point, coordinates):
    xmin = coordinates[_x]
    ymin = coordinates[_y]
    xmax = xmin + coordinates[_w]
    ymax = ymin + coordinates[_h]
    if point[0] >= xmin and point[0] <= xmax and point[1] >= ymin and point[1] <= ymax:
        return True
    return False


def main():
    (coordinates,img) = symSeg(image, printOP = True)
    graph, data = LOSGraphBuilder(coordinates, img)

    LOSViewer(coordinates, graph)

    for i in range(len(data[0])):
        print(data[0,i], data[1,i])
    for i in range(len(coordinates)):
        (cX, cY) = coordinates[i][_cX], coordinates[i][_cY]
        img = cv2.putText(img, str(i), (int(cX), int(cY)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),thickness=1)
    cv2.imshow('nodes', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()