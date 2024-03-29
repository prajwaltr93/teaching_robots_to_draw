#filename : simple_search.py
#author : PRAJWAL T R
#date last modified : Mon SEPT 04 11:09:55 2020
#comments :

'''
    simpleSearch performs a DFS search to find connected strokes, which is useful of stroke segmentation
    and obtaining Lcon for local model while running in real time, given the start pixel and size of image(search space)
'''
# imports
import copy as cp
import numpy as np

stack = []
visited = {}

# Constants
PROCESSING = 1
VISITED = 2
UNVISITED = 0
threshold = 0

def getSliceWindow(current_xy):
    # cordinates for slicing 3*3 window
    slice_begin = [0, 0] # pass by reference sucks some times
    slice_begin[1] = current_xy[1] - 1
    slice_begin[0] = current_xy[0] - 1
    return slice_begin

def findAdjacent(current_xy, img, threshold):
    neighbourers = []
    # changes made here stays here !
    img = cp.deepcopy(img)
    # get search window
    slice_begin =  getSliceWindow(current_xy)
    img[current_xy[1], current_xy[0]] = 0 # exclude current point from searching
    img_slice = img[slice_begin[1] : slice_begin[1] + 3, slice_begin[0] : slice_begin[0] + 3]
    # get cordinates of neighbourers above threshold
    cord = np.where(img_slice > threshold)
    # offset withing search window
    delta = [[i - 1, j - 1] for i, j in zip(cord[1], cord[0])] # inverted axis x,y = y,x
    # add delta to current_xy to get global coordinate
    for d in delta:
        neighbourers.append([current_xy[0] + d[0], current_xy[1] + d[1]])
    return neighbourers

def simpleSearch(start, img, size):

    connected_points = []

    # check if start cordinate is false start
    if(img[start[1]][start[0]] == 0):
        # no possible connected stroke can be found
        connected_points.append(list(start))
        return connected_points

    # prepare visited structure
    for i in range(size):
        for j in range(size):
            visited[[i, j].__repr__()] = UNVISITED

    visited[list(start).__repr__()] = PROCESSING
    stack.append(list(start))

    # DFS search the image for connected strokes
    while(len(stack) != 0):
        current_xy = stack.pop()
        visited[current_xy.__repr__()] = VISITED
        connected_points.append(current_xy)
        neighbourers =  findAdjacent(current_xy, img, threshold)
        for neighbour in neighbourers:
            if visited[neighbour.__repr__()] ==  UNVISITED:
                visited[neighbour.__repr__()] = PROCESSING
                stack.append(neighbour)

    # search complete
    return connected_points

# test working of algorithm
if __name__ == "__main__":
    import cv2 as cv
    from skimage.morphology import skeletonize
    import matplotlib.pyplot as plt
    file = "./test_dir/kanji_samples/0f9a8.png"
    thresh_val = 127
    thresh = cv.THRESH_BINARY_INV
    img = cv.imread(file) # 2 - RBG -> GRAYSCALE
    # img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
    # blur image to reduce noise
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.GaussianBlur(img,(3,3),0) # 1 * 1 kernel
    _, img = cv.threshold(img, thresh_val, 255, thresh)
    img[np.where(img > 0)] = 1
    img = skeletonize(img, method='lee')
    # plt.imshow(img, cmap = "Greys_r")
    plt.savefig("./test_dir/image.png")
    plt.close()
    start = (10, 47) # this is given by global model during run time
    points = simpleSearch(list(start), img, img.shape[0])
    draw_img = np.zeros(img.shape)
    # mark points obtained by search algorithm, which are connected
    for point in points:
        draw_img[point[1], point[0]] = 1
    # plt.imshow(draw_img, cmap = "Greys_r")
    plt.savefig("./test_dir/connected.png")
    plt.close()
