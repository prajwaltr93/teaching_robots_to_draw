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

def driver_code(start_xy, end_xy, img):
    connected_points = []
    # prepare visited structure
    visited = {}
    size = img.shape[0]
    for i in range(size):
        for j in range(size):
            visited[[i, j].__repr__()] = UNVISITED

    visited[list(start_xy).__repr__()] = VISITED
    connected_points.append(list(start_xy))
    found = False
    def findShortestPath(start_xy, end_xy, img): # default mutable argument, persistent accross calls
        adjs = findAdjacent(start_xy, img, 0)
        nonlocal found
        for point in adjs:
            if visited[list(point).__repr__()] == UNVISITED and not found:
                if point == end_xy:
                    connected_points.append(point)
                    found = True
                    return connected_points
                else:
                    connected_points.append(point)
                    visited[list(start_xy).__repr__()] = PROCESSING
                    findShortestPath(point, end_xy, img)
                    # visited[list(start_xy).__repr__()] = UNVISITED
                    if not found:
                        connected_points.pop()
        return connected_points
    return findShortestPath(start_xy, end_xy, img)
# test working of algorithm
if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        if argv[1] == 'f':
            import cv2 as cv
            from skimage.morphology import skeletonize
            import matplotlib.pyplot as plt
            file = "./test_dir/kanji_samples/x_stroke.jpg"
            img = cv.imread(file, 0)
            img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
            # TODO : apply blur ex : medianBlur(img, 5)
            # load generated image and apply image transformations
            thresh = cv.THRESH_BINARY_INV
            _, img = cv.threshold(img, 0, 255, thresh + cv.THRESH_OTSU) # use otsu to find appropriate threshold
            img[np.where(img > 0)] = 1 # convert to image with 0's and 1's ex : binary image
            img = skeletonize(img, method="lee")
            stop = [58, 56]
            start = [99, 39]
            points_ = simpleSearch(start, img, img.shape[0])
            assert start in points_
            assert stop in points_
            assert [40, 38]
            points = driver_code(start, stop, img)
            print(points)
            draw_img = np.zeros(img.shape)
            # mark points obtained []by search algorithm, which are connected
            for point in points:
                draw_img[point[1], point[0]] = 1
            # plt.imshow(draw_img, cmap = "Greys_r")
            plt.imshow(draw_img)
            plt.show()
        else:
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
