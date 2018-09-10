import matplotlib.pylab as plt
import numpy as np
import sys
from scipy import misc



def computeH(t1, t2):
    left_list = np.load(t1)
    right_list = np.load(t2)

    if (len(left_list) == 2):
        left_list = np.transpose(left_list)
    right_list = np.load(t2)
    if (len(right_list) == 2):
        right_list = np.transpose(right_list)
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

