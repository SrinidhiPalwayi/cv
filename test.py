import matplotlib.pylab as plt
import numpy as np
import sys
from scipy import misc
#import cv2

""""
run script twice
change the list to the other image and another text file
"""
"""""

right_file = "crop1.png"
left_file = "crop2.png"
r_img = plt.imread(right_file)
l_img = plt.imread(left_file)
f,a = plt.subplots(1,2, figsize=(15,10))
a[0].imshow(l_img)
a[1].imshow(r_img)
pos_right = []
pos_left = []
def onclick(event):
    pos_right.append([event.xdata,event.ydata])
f.canvas.mpl_connect('button_press_event', onclick)
f.show()
plt.show()
with open('right.txt', 'w') as f:
    for item in pos_right:
        f.write("%s\n" % [int(item[0]), int(item[1])])
print(pos_right)

"""""
import json

new_img_size = 2.0
scaler = lambda t,orig_img_size : t * (new_img_size / float(orig_img_size))



def computeH():
    left_list = np.load('cc1.npy')
    right_list = np.load('cc2.npy')
    n = len(left_list)

    left_list_one = np.insert(left_list, 2, 1, axis=1) #one concatenated to the end of each list
    zero_array = np.zeros(3)

    left_x = np.asarray([item[0] for item in left_list])
    left_y = np.asarray([item[1] for item in left_list])

    right_x = np.asarray([item[0] for item in right_list])
    right_y = np.asarray([item[1] for item in right_list])

    x_multiply = np.multiply(left_x, right_x)
    yx_multiply = np.multiply(left_y, right_x)

    xy_multiply = np.multiply(left_x, right_y)
    y_multiply = np.multiply(left_y, right_y)

    final = np.zeros(shape=( 9))
    for i in range(n):
        x_mult= np.asarray([x_multiply[i], yx_multiply[i], right_x[i]])
        top = np.concatenate((-left_list_one[i], zero_array, x_mult), axis=0)
        bot = np.concatenate((zero_array,-left_list_one[i], np.asarray([xy_multiply[i]]),
                              np.asarray([y_multiply[i]]), np.asarray([right_y[i]])), axis =0)
        section = np.vstack((top, bot))
        final = np.vstack((final,section))

    final = final[1:]

    u, s, vh = np.linalg.svd(final)
    H = vh[-1]
    H = np.true_divide(H, H[-1])
    H = np.reshape(H, (3,3))
    return H

def getScaledPoints(textfile):
    f = open(textfile)
    left_list = np.asarray([json.loads(line.strip()) for line in f])
    left_list = np.asarray([[scaler(i) for i in coord] for coord in left_list])
    f.close()
    return left_list


def warpImage(inputIm, refIm, H):
    input = np.asarray(plt.imread(inputIm))
    ref = np.asarray(plt.imread(refIm))
    height, width, channels = input.shape

    print(input)

    top_left = np.asarray([0,0,1]).reshape(3,1)
    top_right = np.asarray([width, 0, 1]).reshape(3,1)

    bot_left = np.asarray([0,height, 1]).reshape(3,1)
    bot_right = np.asarray([width, height, 1]).reshape(3,1)

    corners = np.concatenate((top_left, top_right, bot_left, bot_right), axis=1)
    forward_warped = np.dot(H, corners)

    x_max=y_max = -sys.maxint
    x_min=y_min = sys.maxint

    scale_warped = list()
    for col in forward_warped.T:
        row = np.true_divide(col, col[2])
        if (row[0] > x_max):
            x_max = row[0]
        elif (row[0] < x_min):
            x_min = row[0]
        if (row[1] > y_max):
            y_max = row[1]
        if (row[1] < y_min):
            y_min = row[1]
        scale_warped.append(row)
    scale_warped = np.asarray(scale_warped)
    scale_warped = np.transpose(scale_warped)
    #print(scale_warped)

    y_length = int(np.ceil(y_max)) - int(np.floor(y_min))
    x_length = int(np.ceil(x_max)) - int(np.floor(x_min))
    A = np.asarray([[w, h, 1] for h in range(int(np.floor(y_min)),  int(np.ceil(y_max)))
                    for w in range(int(np.floor(x_min)),  int(np.ceil(x_max)))])
    y_min_ind = int(np.floor(y_min))
    x_min_ind = int(np.floor(x_min))
    H_inv = np.linalg.inv(H)
    size = (y_length, x_length, 3)
    final_image = np.zeros(size)
    mosaic_image = np.zeros(size)
    for h in range(int(np.floor(y_min)),  int(np.ceil(y_max))):
        for w in range(int(np.floor(x_min)),  int(np.ceil(x_max))):
            col = np.reshape([w, h, 1], (3,1))
            on_input = np.matmul(H_inv, col)
            on_input = np.true_divide(on_input, on_input[2])
            y_new = int(round(on_input[1][0]))
            x_new = int(round(on_input[0][0]))
            if(y_new >= 0 and y_new < height):
                #is it in the range of y
                if (x_new >= 0 and x_new < width):
                    #is it in the range of x
                    final_image[h-y_min_ind][w-x_min_ind] = input[y_new][x_new]
                    mosaic_image[h - y_min_ind][w - x_min_ind] = input[y_new][x_new]
            if(h >= 0 and h < ref.shape[0] and w >=0 and w < ref.shape[1]):
                mosaic_image[h-y_min_ind][w-x_min_ind] = ref[h][w]

    warped_image = np.array(final_image, dtype='uint8')
    mosaic = np.array(mosaic_image, dtype='uint8')
    x_len = len(ref[0])
    zero_ref = np.zeros((-y_min_ind, x_len,3))
    print(ref.shape)
    print(zero_ref.shape)

    new = np.concatenate((zero_ref,ref), axis=0)
    show = np.array(new, dtype='uint8')

    zero_side = np.zeros((len(ref), -x_min_ind, 3))

    plt.imshow(mosaic, interpolation='nearest')
    plt.show()

""""

    A = np.transpose(A)
    H_inv = np.linalg.inv(H)
    final_image = np.zeros(y_length, x_length)
    scale_inverse_warp = list()
    for col in inverse_warp:
        row = np.true_divide(col, col[2])
        final_image
        scale_inverse_warp.append(row)
    scale_inverse_warp = np.asarray(scale_inverse_warp)
    scale_inverse_warp = np.transpose(scale_inverse_warp)





    transformed = np.dot(H, np.transpose(A))



    scale_w = list()
    x_max = -sys.maxint
    x_max_y = 0
    x_min = sys.maxint
    x_min_y = 0

    y_max = -sys.maxint
    y_max_x = 0
    y_min = sys.maxint
    y_min_x =0
    for col in transformed.T:
        row = np.true_divide(col, col[2])
        if(row[0] > x_max):
            x_max = row[0]
            x_max_y = row[1]
        elif(row[0] < x_min):
            x_min = row[0]
            x_min_y = row[1]
        if(row[1] > y_max):
            y_max_x = row[0]
            y_max = row[1]
        if(row[1] < y_min):
            y_min_x = row[0]
            y_min = row[1]
        scale_w.append(row)
    print(len(scale_w), len(scale_w[0]))
    print(x_min, x_min_y, x_max, x_max_y, y_min, y_min_x, y_max, y_max_x)
"""


#one_list= np.ones((1,n), dtype=int)
#left_side = np.concatenate((np.transpose(left_list), one_list), axis=0)

#transformed = np.dot(H, left_side)

#print (vh.shape)
#print (vh.shape)
#print(A.shape)
#print(H)

#print(right_list)
#print(computeH())
print(warpImage("crop1.jpg", "crop2.jpg", computeH()))