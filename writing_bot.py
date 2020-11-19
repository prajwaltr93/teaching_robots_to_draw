# imports
import matplotlib.pyplot as plt
import numpy as np
from sys import path
import cv2 as cv
from skimage.morphology import skeletonize

# user-defined module imports
# from local_model import getLocalModel
from global_model import getGlobalModel
kanjivg_modules_path = "../kanjivg_dataset/"
path.append(kanjivg_modules_path)
from drawing_utils import HEIGHT, WIDTH

# constants
X_target_file = "test_dir/kanji_samples/0ff11.svg"
fig, main_axs = plt.subplots(1, 3) # 0->X_target, 1->X_diff, 2-> another two subplots for local and global prediction
_, sec_axs = main_axs.subplots(2, 1) # stack vertically
# create two more subplots on axs 2 of main plt
sec_axs[0].set_title("Global Model prediction")
sec_axs[0].text(0.5, 0.5, "None")
sec_axs[1].set_title("Local Model prediction")
sec_axs[1].text(0.5, 0.5, "None")
# counter to track steps for saving plotter figures
counter = 0

g_line = "G %d %d"
l_line = "L %d %d"

# utility functions
def prepImage(file):
    '''
        get .png image, if svg file specified then convert image to ong and apply skeletonization
    '''
    from os import path

    img = np.zeros((HEIGHT, WIDTH))
    if not path.isfile(file.split(".")[0] + ".png"):
        # given svg file, get a png file
        print("INFO : TARGET PNG IMAGE NOT FOUND, CREATING ...")
        from svglib.svglib import svg2rlg
        drawing = svg2rlg(file)
        from reportlab.graphics import renderPM
        renderPM.drawToFile(drawing, file.split(".")[0] + ".png", fmt="PNG")

    img = cv.imread(file.split(".")[0] + ".png", 0) # get GRAYSCALE image
    img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
    # TODO : apply blur ex : medianBlur(img, 5)
    # load generated image and apply image transformations
    thresh = cv.THRESH_BINARY_INV
    _, img = cv.threshold(img, 0, 255, thresh + cv.THRESH_OTSU) # use otsu to find appropriate threshold
    img[np.where(img > 0)] = 1 # convert to image with 0's and 1's ex : binary image
    img = skeletonize(img, method="lee") # convert width of stroke to one pixel wide

    return img

def prepGlobalInput(X_env, X_diff, X_last, X_loc):
    '''
        make input compatible by expanding dimensions along axis = 0
    '''
    return np.expand_dims(np.dstack((X_loc, X_env, X_last, X_diff)), axis=0)

def updateStepCanvas(X_diff, g_predict=None, l_predict=None):
    '''
        update main_axs plt with actions predicted both global and local model
    '''
    global main_axs, sec_axs, counter
    main_axs.axs[1].imhow(X_diff, cmap="Greys_r") # udpate X_diff
    if g_predict:
        sec_axs[0].text(0.5, 0.5, repr(g_predict))
    if l_predict:
        sec_axs[1].text(0.5, 0.5, repr(l_predict))
    # save plotted figure
    plt.savefig("step : " + count.__str__() + ".png")
    counter += 1

# main functions
def globalModelPredict(X_env, X_diff, X_last, X_loc):
    '''
        predict global actions and pass predicted x,y as current_xy to local model
    '''
    next_xy = global_model.predict(prepGlobalInput(X_env, X_diff, X_last, X_loc))
    updateStepCanvas(X_diff, g_predict=next_xy, l_predict=None) # update step made by global model
    print(next_xy)
    exit(0)
    # prep input to local model
    # connected_points = simple_search(next)
def enterSimulation(X_target):
    '''
        start simulation of drawing X_target
    '''
    global main_axs
    # prepare all input to global model ->  X_loc, X_env, X_diff, X_last
    X_diff = cp.deepcopy(X_target)
    X_env = np.zeros((HEIGHT, WIDTH))
    X_last = np.zeros((HEIGHT, WIDTH))
    X_loc = np.zeros((HEIGHT, WIDTH))
    # update canvas state
    main_axs[0].imshow(X_target, cmap="Greys_r") # X_target
    globalModelPredict(X_env, X_diff, X_last, X_loc)

# start control from here
if __name__ == "__main__":
    # get image X_target
    X_target = prepImage(file=X_target_file)
    # enter controlled simulation
    enterSimulation(X_target=X_target)
