#filename : writing_bot.py
#author : PRAJWAL T R
#date last modified : Sat Jan  9 23:44:40 2021
#comments :
'''
    Implement full global and local model control cycle for reproducing written image, save intermediataries in plots
'''

# imports
import numpy as np
import copy as cp
import cv2 as cv
from sys import path
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# user-defined module imports
from local_model import getLocalModel
from global_model import getGlobalModel
kanjivg_modules_path = "../kanjivg_dataset/"
path.append(kanjivg_modules_path)
from drawing_utils import HEIGHT, WIDTH, highlightPoints
from simple_search import simpleSearch, driver_code, findAdjacent

# variable to keep track of steps
counter = 0
local_step_plt_path = "./test_dir/sim_steps/"
# define global and local model with trained weights loaded
global_model = getGlobalModel()
local_model =  getLocalModel()

# line formats for local and global model prediction printing
g_line = "G %d %d"
l_line = "L %d %d"
# touch thresh to change control between local and global model
touch_thresh = 1e-1

def prepImage(file):
    '''
        get .png image, if svg file specified then convert image to ong and apply skeletonization
    '''
    from os import path

    img = np.zeros((HEIGHT, WIDTH))
    if path.isfile(file.split(".")[0] + ".svg"):
        # given svg file, get a png file
        print("INFO : TARGET PNG IMAGE NOT FOUND, CREATING ...")
        from svglib.svglib import svg2rlg
        drawing = svg2rlg(file.split(".")[0] + ".svg")
        from reportlab.graphics import renderPM
        renderPM.drawToFile(drawing, file.split(".")[0] + ".png", fmt="PNG")
        img = cv.imread(file.split(".")[0] + ".png", 0)

    elif path.isfile(file.split(".")[0] + ".png"):
        img = cv.imread(file.split(".")[0] + ".png", 0) # get GRAYSCALE image
    else:
        img = cv.imread(file.split(".")[0] + ".jpg", 0)
    print("[INFO] : IMAGE READ")
    img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
    # TODO : apply blur ex : medianBlur(img, 5)
    # load generated image and apply image transformations
    thresh = cv.THRESH_BINARY_INV
    _, img_ = cv.threshold(img, 0, 255, thresh + cv.THRESH_OTSU) # use otsu to find appropriate threshold
    img_[np.where(img_ > 0)] = 1 # convert to image with 0's and 1's ex : binary image
    print("[INFO] : APPLYING SKELETONIZATION")
    img_ = skeletonize(img_, method="lee") # convert width of stroke to one pixel wide
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap="Greys_r")
    axs[1].imshow(img_, cmap="Greys_r")
    axs[0].set_title("Original Image")
    axs[1].set_title("Processed Image")
    plt.show()
    return img_

def updateCanvas(env_img, diff_img, con_img, next_xy_grid):
    '''
        save each instances/steps taken by local model in plots
    '''
    global counter
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(env_img, cmap="Greys_r")
    axs[1].imshow(diff_img, cmap="Greys_r")
    axs[2].imshow(con_img, cmap="Greys_r")
    # axs[3].imshow(np.reshape(next_xy_grid, (5, 5)), cmap="Greys_r")
    # update legend
    axs[0].set_title("env_img")
    axs[1].set_title("diff_img")
    axs[2].set_title("con_img")
    # axs[3].set_title("next_xy pred")
    plt.savefig(local_step_plt_path + "local step : " + counter.__str__() + ".png")
    plt.close()
    counter += 1

def getSliceWindow(current_xy):
    '''
        generate two variables begin and size for dynamice tensor slicing using tf.slice
    '''
    x, y = current_xy[0], current_xy[1]
    begin = [y - 2, x - 2 , 0] # zero slice begin channel dimension
    return np.array(begin)

def _getSliceWindow(slice_begin):
    '''
        revert slice window transform
    '''
    x, y = slice_begin[0], slice_begin[1]
    restore = [y + 2, x + 2] # restore transformed cordinates
    return np.array(restore)

def get2DCordinatesGlobal(index):
    '''
        convert index of flattened array to index of 2d array ex : 100 * 100 image
    '''
    return (index%HEIGHT, index//WIDTH)

def get2DCordinatesLocal(index):
    # width and height of cropped image
    crop_hw = 5
    # returns plt cordinates x, y from col, row ex: x --> row  and y --> col
    return (index%crop_hw, index//crop_hw)

def getNextCordinates(slice_begin, next_xy):
    # convert to actual cordinates
    current_xy = [slice_begin[1] + 2, slice_begin[0] + 2]
    delta = [next_xy[0] - 2, next_xy[1] - 2]
    return [current_xy[0] + delta[0], current_xy[1] + delta[1]] # actual next_xy

def prepGlobalInput(X_env, X_diff, X_last, X_loc):
    '''
        make input compatible by expanding dimensions along axis = 0
    '''
    return np.expand_dims(np.dstack((X_loc, X_env, X_last, X_diff)), axis=0)

def prepLocalInput(inpts):
    # make inputs suitable to be taken as input to local model
    return [np.expand_dims(np.dstack((inpts[0], inpts[1], inpts[2])), axis=0), np.expand_dims(inpts[3], axis=0)] # [x1, x2]

def checkPrediction(next_xy, connected_points):
    '''
        check if local model predicted correctly based on some logic
    '''
    if next_xy in connected_points[1:3]: # check within two points of connected_points
        return True
    else: # correct prediction, modify next_xy
        print("INFO : PREDICTION MADE : ", next_xy)
        # side effect
        next_xy[0], next_xy[1] = connected_points[1][0], connected_points[1][1] # choose next immediate point
        return False

def local_model_predict(slice_begin, connected_points, env_img, diff_img, con_img, start_xy):
    '''
        recursive function to predict local actions
    '''
    touch_pred, next_xy_pred = local_model.predict(prepLocalInput([env_img, diff_img, con_img, slice_begin]))
    updateCanvas(env_img, diff_img, con_img, next_xy_pred)
    if np.argmax(next_xy_pred) != 12 : # for touch < touch thresh, next_xy_pred = 12 or (2,2) in 2D matrix
        next_xy = get2DCordinatesLocal(np.argmax(next_xy_pred))
        next_xy = getNextCordinates(slice_begin, next_xy)
        print(next_xy, touch_pred, connected_points, start_xy)
        # assume local model predicts accurately
        # next_xy in connected_points
        if next_xy not in connected_points:
            next_xy = connected_points[0]
        print(l_line % tuple(next_xy))
        stroke = driver_code(start_xy, next_xy, con_img)
        print(stroke)
        for point in stroke:
            env_img[point[1], point[0]] = 1 # write
            diff_img[point[1], point[0]] = 0 # erase

        # update remaining connected points
        # TODO : remove all point from stroke in connected_points except last
        stroke = stroke[:-1] if len(stroke) > 1 else stroke
        for point in stroke:
            try:
                index_ = connected_points.index(point) # .remove ?
                connected_points.pop(index_)
            except:
                pass
        # connected_points = connected_points[connected_points.index(next_xy) :] # inclusive of current point
        slice_begin = getSliceWindow(next_xy)
        start_xy = next_xy
        return (local_model_predict(slice_begin, connected_points, env_img, diff_img, con_img, start_xy))
    return _getSliceWindow(slice_begin)

def globalModelPredict(X_env, X_diff, X_last, X_loc):
    '''
        predict global actions and pass predicted x,y as current_xy to local model
    '''
    next_xy = global_model.predict(prepGlobalInput(X_env, X_diff, X_last, X_loc))
    next_xy = get2DCordinatesGlobal(np.argmax(next_xy[0]))
    print(g_line % next_xy)
    connected_points = simpleSearch(start=next_xy, img=X_diff, size=HEIGHT)
    if len(connected_points) == 0:
        print("ERROR : GLOBAL MODEL MADE FALSE PREDICTION")
        return
    if (len(connected_points) == 1):
        # TODO : FIX BUG CREATING NOISE IN CON_IMAGE
        return

    # prep input to local model
    X_con = highlightPoints(connected_points)
    slice_begin = getSliceWindow(next_xy)
    X_cord = local_model_predict(slice_begin, connected_points, X_env, X_diff, X_con, start_xy=next_xy)
    X_loc = np.zeros((HEIGHT, WIDTH))
    X_loc[X_cord[1], X_cord[0]] = 1
    globalModelPredict(X_env, X_diff, X_con, X_loc)
    # TODO : call global model predict again

def enterSimulation(X_target):
    '''
        diff_img start simulation of drawing X_target
    '''
    # prepare all input to global model ->  X_loc, X_env, X_diff, X_last
    X_diff = cp.deepcopy(X_target)
    X_env = np.zeros((HEIGHT, WIDTH))
    X_last = np.zeros((HEIGHT, WIDTH))
    X_loc = np.zeros((HEIGHT, WIDTH))
    # update canvas state
    globalModelPredict(X_env, X_diff, X_last, X_loc)

# start control from here
if __name__ == "__main__":
    # get image X_target
    samples_path = "test_dir/kanji_samples/"
    from sys import argv
    from os import path
    if len(argv) == 2:
        file = argv[1]
        file = samples_path + file
        if (path.isfile(file)):
            X_target = prepImage(file=file)
            # enter controlled simulation
            enterSimulation(X_target=X_target)
            # clean up sim steps
            print("[INFO] : clean up sim steps ?(y/n)")
            i = input()
            if i.lower() == 'n':
                pass
            else:
                from os import walk, remove
                try:
                    _, _, files = next(walk("./test_dir/sim_steps/"))
                    print(f"[INFO] : removing {len(files)} simulation files")
                    for file in files:
                        remove("./test_dir/sim_steps/" + file)
                except:
                    print("[INFO] : no files found")
        else:
            print(f"[ERROR] : File {file} Does not Exist")
    else:
        print("[ERROR] : Enter file name")
