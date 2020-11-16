#filename : test_local_model.py
#author : PRAJWAL T R
#date last modified : Wed Nov  4 09:24:36 2020
#comments :

'''
    test working of local model by pseudo input from global model.
'''

# imports
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sys import path
from skimage.morphology import skeletonize

from simple_search import simpleSearch
drawing_utils_path = "../kanjivg_dataset/"
path.append(drawing_utils_path)
from drawing_utils import HEIGHT, WIDTH, drawFromPoints
from drawing_utils import getStrokesIndices, parsePointString, drawStroke, highlightPoints
from local_model import getLocalModel

# constants and magic numbers
local_model = getLocalModel()
# thresh value to flip pixels
thresh_val = 137
# touch thresh to change control between local and global model
touch_thresh = 0.3
# variable to keep track of steps
counter = 0
local_step_plt_path = "./test_dir/local_steps/"

# utility functions
def get2DCordinates(index):
    # width and height of cropped image
    crop_hw = 5
    # returns plt cordinates x, y from col, row ex: x --> row  and y --> col
    return (index%crop_hw, index//crop_hw)

def getNextCordinates(slice_begin, next_xy):
    # convert to actual cordinates
    current_xy = [slice_begin[1] + 2, slice_begin[0] + 2]
    delta = [next_xy[0] - 2, next_xy[1] - 2]
    return (current_xy[0] + delta[0], current_xy[1] + delta[1]) # actual next_xy

def pseudoGlobalModelGenerator(character_file):
    '''
        returns a proxy generator which yields outputs of global model
    '''
    svg_string = open(character_file, "r").read()
    X_target, m_indices = getStrokesIndices(svg_string)
    # return genrator with starting point of strokes
    return (parsePointString(X_target[m_ind]) for m_ind in m_indices)

def getSliceWindow(current_xy):
    '''
        generate two variables begin and size for dynamice tensor slicing using tf.slice
    '''
    x, y = current_xy[0], current_xy[1]
    begin = [y - 2, x - 2 , 0] # zero slice begin channel dimension
    return np.array(begin)

def prepImage(file, flag):
    '''
        get target image
        for testing only :
        flag :
        1 -> from svg : write image from svg data
        0 -> from png : apply threshold and skeletonise image
    '''
    from os import path
    img = np.zeros((HEIGHT, WIDTH))
    if not flag:
        # TODO: FIX skeletonize
        # get target image from png
        print("INFO : TARGET IMAGE FROM PNG")
        if not path.isfile(file.split(".")[0] + ".png"):
            # get a png file from svg file
            print("INFO : TARGET PNG IMAGE NOT FOUND, CREATING ...")
            from svglib.svglib import svg2rlg
            drawing = svg2rlg(file)
            from reportlab.graphics import renderPM
            renderPM.drawToFile(drawing, file.split(".")[0] + ".png", fmt="PNG")
        img = cv.imread(file.split(".")[0] + ".png")
        img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
        # load generated image and apply image transformations
        thresh = cv.THRESH_BINARY_INV
        # img = cv.imread('./res/japanese.png') # 2 - RBG -> GRAYSCALE
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, img = cv.threshold(img, thresh_val, 255, thresh)
        img[np.where(img > 0)] = 1 # convert to image with 0's and 1's ex : binary image
        img = skeletonize(img, method="lee") # convert width of stroke to one pixel wide
    else:
        # get target image from svg
        print("INFO : TARGET IMAGE FROM SVG")
        X_target, m_indices = getStrokesIndices(open(file, 'r').read())
        img = drawStroke(X_target)

    return img
def prepInput(inpts):
    # make inputs suitable to be taken as input to model
    return [np.expand_dims(np.dstack((inpts[0], inpts[1], inpts[2])), axis=0), np.expand_dims(inpts[3], axis=0)] # [x1, x2]

def updateCanvas(env_img, diff_img, con_img, next_xy_grid):
    '''
        save each instances/steps taken by local model in plots
    '''
    global counter
    _, axs = plt.subplots(1, 4)
    axs[0].imshow(env_img, cmap="Greys_r")
    axs[1].imshow(diff_img, cmap="Greys_r")
    axs[2].imshow(con_img, cmap="Greys_r")
    axs[3].imshow(np.reshape(next_xy_grid, (5, 5)), cmap="Greys_r")
    # update legend
    axs[0].set_title("env_img")
    axs[1].set_title("diff_img")
    axs[2].set_title("con_img")
    axs[3].set_title("next_xy pred")
    plt.savefig(local_step_plt_path + "local step : " + counter.__str__() + ".png")
    plt.close()
    counter += 1

def local_model_predict(slice_begin, connected_points, env_img, diff_img, con_img):
    touch_pred, next_xy_pred = local_model.predict(prepInput([env_img, diff_img, con_img, slice_begin]))
    updateCanvas(env_img, diff_img, con_img, next_xy_pred)
    print(touch_pred)
    if touch_pred[0] >= touch_thresh:
        if len(connected_points) == 0:
            return
        next_xy = get2DCordinates(np.argmax(next_xy_pred))
        next_xy = getNextCordinates(slice_begin, next_xy)
        if list(next_xy) in connected_points:
            print('SUCCESS : POINT PREDICTED')
            print("INFO : PREDICTION MADE : ", next_xy)
        else:
            # local model prediction failed, use a point from connected points to help
            print("FAILED : POINT PREDICTED")
            print("INFO : PREDICTION MADE : ", next_xy)
            # choose next point in connected_points
            try:
                # help local model for next_xy
                next_xy = connected_points[1]
            except:
                next_xy = connected_points[0]

        # update input variables to local model
        stroke = connected_points[0 : connected_points.index(list(next_xy))+1]
        for point in stroke:
            env_img[point[1], point[0]] = 1 # write
            diff_img[point[1], point[0]] = 0 # erase

        # update remaining connected points
        connected_points = connected_points[connected_points.index(list(next_xy)) + 1 : ]
        slice_begin = getSliceWindow(next_xy)
        local_model_predict(slice_begin, connected_points, env_img, diff_img, con_img)

if __name__ == "__main__":
    file = "test_dir/kanji_samples/02ea8.svg"
    # get X_target from png image
    X_target_img = prepImage(file, flag = 1)
    # get current_xy
    stroke = 2 # nth stroke
    global_model_predict = pseudoGlobalModelGenerator(file)
    for i in range(stroke):
        current_xy = next(global_model_predict)
    connected_points = simpleSearch(current_xy, X_target_img, X_target_img.shape[0])[:] # remove first point from prediction
    # prepare input to local model, env_img, diff_img, con_img, ext (current_xy)
    env_img = np.zeros((HEIGHT, WIDTH))
    diff_img = X_target_img
    con_img = highlightPoints(connected_points)
    slice_begin = getSliceWindow(current_xy)
    print(connected_points)
    local_model_predict(slice_begin, connected_points, env_img, diff_img, con_img)
