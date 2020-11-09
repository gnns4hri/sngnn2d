import sys
import random
import json
import numpy as np
import os
import cv2
from graph2imageUp import *
import time
import math

def toColour(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def beautify_grey_image(img):
    return beautify_image(toColour(img))

def beautify_image(img):
    def convert(value):
        colors = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]]
        v = (255 - value) / 255
        if v >= 1:
            idx1 = 3
            idx2 = 3
            fract = 0
        else:
            v = v * 3
            idx1 = math.floor(v)
            idx2 = idx1 + 1
            fract = v - idx1
        r = (colors[idx2][0] - colors[idx1][0]) * fract + colors[idx1][0]
        g = (colors[idx2][1] - colors[idx1][1]) * fract + colors[idx1][1]
        b = (colors[idx2][2] - colors[idx1][2]) * fract + colors[idx1][2]
        red = r * 255
        green = g * 255
        blue = b * 255
        return red, green, blue
    for row in range(0,img.shape[0]):
        for col in range(0, img.shape[1]):
            v = float(img[row, col, 2])
            bad = 255.-v
            red, blue, green = convert(v)
            th = 215.
            img[row, col, 0] = blue
            img[row, col, 1] = green
            img[row, col, 2] = red
    return img

def test_sn(sngnn, scenario):
    ret = sngnn.predict(scenario)/255
    ret = ret.reshape(socnavImg.output_width, socnavImg.output_width)
    ret = cv2.resize(ret, (100, 100), interpolation=cv2.INTER_NEAREST)
    return ret

def test_json(sngnn, filename, line):
    ret = sngnn.predict(filename, line)
    ret = ret.reshape(socnavImg.output_width, socnavImg.output_width)
    return ret


def add_walls_to_grid(image, read_structure):
    # Get the wall points and repeat the first one to close the loop
    walls = read_structure['room']
    walls.append(walls[0])
    # Initialise data
    SOCNAV_AREA_WIDTH = 800.
    grid_rows = image.shape[0]
    grid_cols = image.shape[1]
    # Function doing hte mapping
    def coords_to_ij(y, x):
        px = int((x*grid_cols)/SOCNAV_AREA_WIDTH + grid_cols/2)
        py = int((y*grid_rows)/SOCNAV_AREA_WIDTH + grid_rows/2)
        return px, py
    for idx, w in enumerate(walls[:-1]):
        p1x, p1y = coords_to_ij(w[0], w[1])
        p2x, p2y = coords_to_ij(walls[idx+1][0], walls[idx+1][1])
        cv2.line(image, (p1y, p1x), (p2y, p2x), (50, 50, 50), 2)
        # line(grid_gray, p1, p2, 0, 15, LINE_AA);
    return image




filename = 'out_test.json'
#filename = 'socnav_test.json'

structures = open('data/'+filename, 'r').readlines()
ts = []
counter = -1
count = 0


device = 'cpu'
if 'cuda' in sys.argv:
    device = 'cuda'

while True:
    sngnn = SNGNN2D('./', device)
    t = os.path.getmtime('SNGNN2D.tch') 

    while t == os.path.getmtime('SNGNN2D.tch'):
        counter += 1
        structure = structures[counter]
        line = counter
        read_structure = json.loads(structure)
        identifier = read_structure['identifier']

        label_filename = 'labels/all' + '/' + identifier + '.png'
        label = cv2.imread(label_filename)
        if label is None:
            print('Couldn\'t read label file', label_filename)
            continue

        image_filename = 'images/all' + '/' + identifier + '.png'
        image = cv2.imread(image_filename)
        if image is None:
            print('Couldn\'t read image file:', image_filename)
            continue

        time_a = time.time()
        ret1 = test_json(sngnn, filename, line)
        time_b = time.time()
        tt = time_b-time_a
        print('Time:', tt)

        icc, jcc = int(ret1.shape[0]/2), int(ret1.shape[1]/2)
        ret1 = (255.*ret1).astype(np.uint8)
        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)#.astype(np.uint8)
        label = cv2.resize(label, (300, 300), interpolation=cv2.INTER_CUBIC)#.astype(np.uint8)/255
        ret1  = cv2.resize(ret1,  (300, 300), interpolation=cv2.INTER_CUBIC)#/255

        if label is None:
            print('Couldn\'t read label file', label_filename)
            continue
        count += 1

        time_show = time.time()
        label = add_walls_to_grid(label.astype(np.uint8), read_structure)
        finalo = add_walls_to_grid(toColour((ret1).astype(np.uint8)), read_structure)
        ret = np.concatenate((
            image,
            #beautify_image((label).astype(np.uint8)),
            label,
            #beautify_grey_image((ret1*255).astype(np.uint8))),
            finalo),
            axis=1)
        #cv2.imwrite('frame'+str(count)+'.png', ret)

        while True:
            cv2.imshow("SNGNN2D", ret)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                sys.exit()
            else:
                if time.time()-time_show > 4.:
                    break
    print('Reloading SNGNN2D')
    sngnn = None



