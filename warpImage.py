import numpy as np
import sys
import matplotlib.pylab as plt

def warpImage(inputIm, refIm, H):
    input = np.asarray(plt.imread(inputIm))
    ref = np.asarray(plt.imread(refIm))
    height, width, channels = input.shape

    print(input.shape, "shape ")

    top_left = np.asarray([0, 0, 1]).reshape(3, 1)
    top_right = np.asarray([width, 0, 1]).reshape(3, 1)

    bot_left = np.asarray([0, height, 1]).reshape(3, 1)
    bot_right = np.asarray([width, height, 1]).reshape(3, 1)

    corners = np.concatenate((top_left, top_right, bot_left, bot_right), axis=1)
    forward_warped = np.dot(H, corners)

    x_max = y_max = -sys.maxint
    x_min = y_min = sys.maxint

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

    y_length = int(np.ceil(y_max)) - int(np.floor(y_min))
    x_length = int(np.ceil(x_max)) - int(np.floor(x_min))

    y_min_ind = int(np.floor(y_min))
    x_min_ind = int(np.floor(x_min))
    H_inv = np.linalg.inv(H)
    size = (y_length, x_length, 3)
    final_image = np.zeros(size)
    mosaic_image = np.zeros(size)
    for h in range(int(np.floor(y_min)), int(np.ceil(y_max))):
        for w in range(int(np.floor(x_min)), int(np.ceil(x_max))):
            col = np.reshape([w, h, 1], (3, 1))
            on_input = np.matmul(H_inv, col)
            on_input = np.true_divide(on_input, on_input[2])
            y_new = int(round(on_input[1][0]))
            x_new = int(round(on_input[0][0]))
            if (y_new >= 0 and y_new < height):
                # is it in the range of y
                if (x_new >= 0 and x_new < width):
                    # is it in the range of x
                    final_image[h - y_min_ind][w - x_min_ind] = input[y_new][x_new]
                    mosaic_image[h - y_min_ind][w - x_min_ind] = input[y_new][x_new]
            if (h >= 0 and h < ref.shape[0] and w >= 0 and w < ref.shape[1]):
                mosaic_image[h - y_min_ind][w - x_min_ind] = ref[h][w]

    warpedIm = np.array(final_image, dtype='uint8')
    mergedIm = np.array(mosaic_image, dtype='uint8')
    #i'm not going to the right enogh

    return (warpedIm, mergedIm)