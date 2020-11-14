#filename : test_local_model.py
#author : PRAJWAL T R
#date last modified : Wed Nov  4 09:24:36 2020
#comments :

'''
    test working of local model by pseudo input from global model.
'''

# imports
from local_model import getLocalModel
import sys
import numpy as np
import cv2 as cv
import copy as cp
from simple_search import simpleSearch
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# add drawing_utils module to sys path
drawing_utils_path = "../kanjivg_dataset/"
sys.path.append(drawing_utils_path)
from drawing_utils import HEIGHT, WIDTH, drawFromPoints
from drawing_utils import getStrokesIndices, parsePointString, drawStroke, highlightPoints

# globals and Constants
local_model = getLocalModel()
local_step_plt_path = "./test_dir/local_steps/"
res_file = "test_dir/kanji_samples/0f9b4.svg"
# thresh value to flip pixels
thresh_val = 137
# counter to save plots of local model
counter = 0
# thresh value of touch to decide control transfer beteen local and global model
touch_thresh = 0.0
# temp canvas
temp_canvas = np.zeros((HEIGHT, WIDTH))

# helper functions
def get2DCordinates(index):
    # width and height of cropped image
    crop_hw = 5
    # returns plt cordinates x, y from col, row ex: x --> row  and y --> col
    return (index%crop_hw, index//crop_hw)

def getNextCordinates(current_xy, next_xy):
    # convert to actual cordinates
    delta = [next_xy[0] - 1, next_xy[1] - 1]
    return (current_xy[0] + delta[0], current_xy[1] + delta[1])

def prepInput(inpts):
    # make inputs suitable to be taken as input to model
    return [np.expand_dims(np.dstack((inpts[0], inpts[1], inpts[2])), axis=0), np.expand_dims(np.array(list(inpts[3][::-1]) + [0]), axis=0)] # [x1, x2]

# pseudo global model output generator
def pseudoGlobalModelGenerator(character_file):
    '''
        returns a generator which yeilds outputs of global model
    '''
    svg_string = open(character_file, "r").read()
    X_target, m_indices = getStrokesIndices(svg_string)
    # return genrator with starting poinst of strokes
    return (parsePointString(X_target[m_ind]) for m_ind in m_indices)

def prepImage(file, flag):
    '''
        get target image
        for testing only :
            flag : get handwritten image
                1 -> from svg : write image from svg data
                0 -> from png : apply threshold and skeletonise image
    '''
    from os import path
    if flag:
        print("INFO : TARGET IMAGE FROM SVG")
        X_target, m_indices = getStrokesIndices(open(file, 'r').read())
        img = drawStroke(X_target)
    else:
        print("INFO : TARGET IMAGE FROM PNG")
        print(file)
        if not path.isfile(file.split(".")[0] + ".png"):
            # get a png file from svg file
            print("INFO : TARGET IMAGE NOT FOUND, CREATING")
            from svglib.svglib import svg2rlg
            drawing = svg2rlg(file)
            from reportlab.graphics import renderPM
            renderPM.drawToFile(drawing, file.split(".")[0] + ".png", fmt="PNG")
        print(file.split(".")[0] + ".png")
        img = cv.imread(file.split(".")[0] + ".png")
        img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
        # load generated image and apply image transformations
        thresh = cv.THRESH_BINARY_INV
        # img = cv.imread('./res/japanese.png') # 2 - RBG -> GRAYSCALE
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, img = cv.threshold(img, thresh_val, 255, thresh)
        img[np.where(img > 0)] = 1 # convert to image with 0's and 1's ex : binary image
        img = skeletonize(img, method='lee') # convert width of stroke to one pixel wide
    plt.imshow(img)
    plt.show()
    return img

def updateCanvas(env_img, diff_img, con_img, ext_inp, touch, next_xy_grid):
    '''
        save each instances/steps taken by local model in plots
    '''
    global counter, temp_canvas
    _, axs = plt.subplots(1, 5)
    axs[0].imshow(env_img, cmap="Greys_r")
    axs[1].imshow(diff_img, cmap="Greys_r")
    axs[2].imshow(temp_canvas, cmap="Greys_r")
    axs[3].imshow(con_img, cmap="Greys_r")
    axs[4].imshow(np.reshape(next_xy_grid, (5, 5)), cmap="Greys_r")
    # update legend
    axs[0].set_title("env_img")
    axs[1].set_title("diff_img")
    axs[2].set_title("actual output")
    axs[3].set_title("connected stroke")
    axs[4].set_title("next xy")
    plt.savefig(local_step_plt_path + "local step : " + counter.__str__() + ".png")
    plt.close()
    counter += 1

def local_model_predict(current_xy, connected_points, env_img, diff_img, con_img):
    '''
        recursive function which predicts local actions
    '''
    touch, next_xy_grid = local_model.predict(prepInput([env_img, diff_img, con_img, current_xy]))
    updateCanvas(env_img, diff_img, con_img, current_xy, touch, next_xy_grid)
    if touch[0] >= touch_thresh: # pass control to local model
        if len(connected_points) ==  0:
            # miss predicted touch value
            print("FAILED :  TOUCH PREDICTED")
            print("INFO : PREDICTION MADE : ", touch)
            print("INFO : TESTING ENDS")
            return # testing ends
        # get 2d cordinates of 1d array
        next_xy = get2DCordinates(np.argmax(next_xy_grid))
        next_xy = getNextCordinates(current_xy, next_xy)
        if list(next_xy) in connected_points:
            print('SUCCESS : POINT PREDICTED')
            print("INFO : PREDICTION MADE : ", next_xy)
            cv.line(temp_canvas,tuple(current_xy), tuple(next_xy), 1, 1, cv.LINE_AA) # update temp_canvas -> all local model predictions
            h_next_xy = next_xy
        else:
            print("FAILED : POINT PREDICTED")
            print("INFO : PREDICTION MADE : ", next_xy)
            cv.line(temp_canvas,tuple(current_xy), tuple(next_xy), 1, 1, cv.LINE_AA) # update temp_canvas -> all local model predictions
            try:
                # help local model for next_xy
                h_next_xy = connected_points[1]
            except:
                h_next_xy = connected_points[0]
        # update env, diff images
        stroke = connected_points[0 : connected_points.index(list(h_next_xy))+1]
        for point in stroke:
            env_img[point[1], point[0]] = 1 # write
            diff_img[point[1], point[0]] = 0 # erase

        # update remaining connected points
        connected_points = connected_points[connected_points.index(list(h_next_xy)) + 1 : ]
        local_model_predict(next_xy, connected_points, env_img, diff_img, con_img)
# test local model
if __name__ == "__main__":
    file = res_file
    stroke = 1 # nth stroke
    global_model_predict = pseudoGlobalModelGenerator(file)
    skel_img = prepImage(file, 0)
    for i in range(stroke):
        start = next(global_model_predict) # starting point for local model
    # X_target, start
    connected_points = simpleSearch(start, skel_img, WIDTH)
    if len(connected_points) == 1:
        print("INFO : starting point miss match")
        exit(0)
    print("connected points for start = " + start.__repr__() + ":", connected_points)
    con_img = highlightPoints(connected_points)
    env_img = np.zeros((HEIGHT, WIDTH))
    diff_img = cp.deepcopy(skel_img)
    local_model_predict(start, connected_points, env_img, diff_img, con_img)
