import matplotlib.pylab as plt
import numpy as np
import sys

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
        #getScaledPoints("left.txt")
    right_list =  np.load('cc2.npy')

    #why is the identity matrix diagonal half
        #getScaledPoints("right.txt")

    n = len(left_list)

    left_x = np.asarray([item[0] for item in left_list])
    left_y = np.asarray([item[1] for item in right_list])

    right_x = np.asarray([item[0] for item in right_list])
    transpose_right_x = np.reshape(right_x, (n,1))
    right_y = np.asarray([item[1] for item in right_list])
    transpose_right_y = np.reshape(right_y, (n,1))

    one_array = np.ones((n,1))
    zero_array = np.zeros((n,3), dtype=int)

    x_multiply = np.multiply(-left_x, right_x)
    x_multiply = np.reshape(x_multiply, (n,1))

    yx_multiply = np.multiply(-left_y, right_x)
    yx_multiply = np.reshape(yx_multiply, (n,1))

    xy_multiply = np.multiply(-left_x, right_y)
    xy_multiply = np.reshape(xy_multiply, (n,1))

    y_multiply = np.multiply(-left_y, right_y)
    y_multiply = np.reshape(y_multiply, (n,1))

    top = np.concatenate((left_list,one_array, zero_array, x_multiply, yx_multiply, transpose_right_x), axis=1)
    bottom = np.concatenate((zero_array, left_list, one_array,xy_multiply, y_multiply, transpose_right_y),axis=1 )

    A = np.concatenate((top,bottom), axis=0)

    u, s, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3,3)
    return H

def getScaledPoints(textfile):
    f = open(textfile)
    left_list = np.asarray([json.loads(line.strip()) for line in f])
    left_list = np.asarray([[scaler(i) for i in coord] for coord in left_list])
    f.close()
    return left_list


def warpImage(inputIm, refIm, H):
    #inputIm is left image
    #refIm is the right image
    input = np.asarray(plt.imread(inputIm))
    print(input)
    height, width, channels = input.shape
    print(height, width, channels)

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
    print(scale_warped)
    A = np.asarray([[w, h, 1] for h in range(int(np.floor(y_min)), int(np.ceil(y_max)))
                    for w in range(int(np.floor(x_min)),int(np.ceil(x_max)))])
    A = np.transpose(A)
    H_inv = np.linalg.inv(H)

    inverse_warp = np.dot(H_inv, A)
    scale_inverse_warp = list()
    for col in inverse_warp:
        row = np.true_divide(col, col[2])
        scale_inverse_warp.append(row)
    scale_inverse_warp = np.asarray(scale_inverse_warp)
    scale_inverse_warp = np.transpose




""""
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