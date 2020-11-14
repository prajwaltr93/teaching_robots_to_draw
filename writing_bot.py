from local_model import getLocalModel
import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import copy as cp
from simple_search import simpleSearch

# Constants and magic numbers
crop_hw = 5

# helper functions
def get2DCordinates(index):
    return (index//crop_hw, index%crop_hw) # returns plt cordinates x,y

def getNextCordinates(current_xy, next_xy):
    delta = [next_xy[0] - 1, next_xy[1] - 1]
    return [current_xy[0] + delta[0], current_xy[1] + delta[1]]

def prepInput(inpts):
    return [np.expand_dims(np.dstack((inpts[0], inpts[1], inpts[2])), axis=0), np.expand_dims(np.array(list(inpts[3][::-1]) + [0]), axis=0)] # [x1, x2]

if __name__ == "__main__":
    # test local working of local model
    thresh = cv.THRESH_BINARY_INV
    img = cv.imread('./res/japanese.png') # 2 - RBG -> GRAYSCALE
    img = cv.resize(img, (100, 100), cv.INTER_CUBIC)
    # blur image to reduce noise
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _,img = cv.threshold(img, 160, 255, thresh)
    img[np.where(img > 0)] = 1
    img = skeletonize(img, method='lee')
    plt.imshow(img, cmap = "Greys_r")
    plt.show()
    start = (44, 11) # this is given global model during run time
    connected_points = simpleSearch(list(start), img, img.shape[0])
    local_model_weights_path = "./weights/local_model_weights"
    # test local working of local model
    local_model = getLocalModel()
    env_img = np.zeros((100, 100))
    con_img = cp.deepcopy(img)
    diff_img = cp.deepcopy(con_img)
    print(connected_points)
    def local_model_predict(current_xy, env_img, diff_img, con_img):
        _, axs = plt.subplots(1, 6)
        axs[0].imshow(env_img, cmap="Greys_r")
        axs[1].imshow(diff_img, cmap="Greys_r")
        axs[2].imshow(con_img, cmap="Greys_r")
        axs[3].axis('off')
        axs[3].text(0.0, 0.5, repr(list(current_xy)))
        axs[4].axis('off')
        axs[0].set_title("env_img")
        axs[1].set_title("diff_img")
        axs[2].set_title("con_img")
        axs[3].set_title("ext vector")
        axs[4].set_title("touch")
        axs[5].set_title("cropped_img")
        touch, next_xy_grid = local_model.predict(prepInput([env_img, diff_img, con_img, current_xy]))
        axs[5].imshow(np.reshape(next_xy_grid, (5,5)), cmap = "Greys_r")
        axs[4].text(0.0, 0.5, repr(list(touch)))
        plt.show()
        if touch > 0.3:
            # get 2d cordinates of 1d array
            next_xy = get2DCordinates(np.argmax(next_xy_grid))
            next_xy = getNextCordinates(current_xy, next_xy)
            if next_xy in connected_points:
                print('hurray')
                stroke = connected_points[0 : connected_points.index(next_xy)+1]
                for ind in range(len(stroke) - 1):
                    cv.line(env_img,tuple(stroke[ind]), tuple(stroke[ind+1]), 1, 1, cv.LINE_AA) # write
                    cv.line(diff_img,tuple(stroke[ind]), tuple(stroke[ind+1]), 0, 1, cv.LINE_AA) # erase
                local_model_predict(next_xy, env_img, diff_img, con_img)
            else:
                print(next_xy)
                print("point not found")
                
    local_model_predict(start, env_img, diff_img, con_img)
