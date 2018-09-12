import numpy as np
import sys
import matplotlib.pylab as plt

def warpImage(inputIm, refIm, H):
    input = np.asarray(plt.imread(inputIm))
    ref = np.asarray(plt.imread(refIm))
    height, width, channels = input.shape


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
    y_max_ind = int(np.ceil(y_max))

    x_min_ind = int(np.floor(x_min))
    x_max_ind = int(np.ceil(x_max))

    H_inv = np.linalg.inv(H)

    #this is warped

    if(ref.shape[0] - y_length > 0):
        extra_ref_y = ref.shape[0] - y_length
    else:
        extra_ref_y =0
    if(ref.shape[1] - x_length>0):
        extra_ref_x =ref.shape[1] - x_length
    else:
        extra_ref_x = 0
    print(y_min)
    print(extra_ref_x, extra_ref_y)
    mosaic_size = (y_length+extra_ref_y, x_length+extra_ref_x, 3)
    last_h =-sys.maxint
    last_w = 0
    first_w = sys.maxint
    mosaic_image = np.zeros(mosaic_size)

    size = (y_length, x_length, 3)
    final_image = np.zeros(size)

    if(extra_ref_x == 0 and extra_ref_y ==0):
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
                        mosaic_image[h - y_min_ind + extra_ref_y][w - x_min_ind + extra_ref_x] = input[y_new][x_new]
                if (h >= 0 and h < ref.shape[0] and w >= 0 and w < ref.shape[1]):
                    mosaic_image[h - y_min_ind ][w - x_min_ind] = ref[h][w]
                    last_h = h
                    if(w<first_w):
                        first_w = w
                    if(w > last_w):
                        last_w=w
        #if have some extra image of the reference image, that doens't have anything to mapped to it on the right side
        if (last_w < ref.shape[1]):
            y_max = int(np.ceil(y_max))
            append = ref[:, last_w:, :]
            if (y_length - y_max > 0):
                top_zero = np.zeros((y_length - y_max, append.shape[1], 3))
                append = np.concatenate((top_zero, append), axis=0)
            if (mosaic_image.shape[0] - append.shape[0] > 0):
                bot_zero = np.zeros((mosaic_image.shape[0] - append.shape[0], append.shape[1], 3))
                append = np.concatenate((append, bot_zero), axis=0)
            mosaic_image = np.concatenate((mosaic_image, append), axis=1)

        #extra on the left
        if(first_w > 0):
            y_max = int(np.ceil(y_max))
            append = ref[:, :first_w, :]
            if (y_length - y_max > 0):
                top_zero = np.zeros((y_length - y_max, append.shape[1], 3))
                append = np.concatenate((top_zero, append), axis=0)
            if (mosaic_image.shape[0] - append.shape[0] > 0):
                bot_zero = np.zeros((mosaic_image.shape[0] - append.shape[0], append.shape[1], 3))
                append = np.concatenate((append, bot_zero), axis=0)
            mosaic_image = np.concatenate((append,mosaic_image), axis=1)
    #if one image is contained within the other image
    else:
        height_first = sys.maxint
        width_first = sys.maxint
        for h in range(0, ref.shape[0]):
            for w in range(0, ref.shape[1]):
                col = np.reshape([w, h, 1], (3, 1))
                on_input = np.matmul(H_inv, col)
                on_input = np.true_divide(on_input, on_input[2])
                y_new = int(round(on_input[1][0]))
                x_new = int(round(on_input[0][0]))
                if (h >= 0 and h < ref.shape[0] and w >= 0 and w < ref.shape[1]):
                    mosaic_image[h][w] = ref[h][w]
                    last_h = h
                    if (w > last_w):
                        last_w = w
                if (y_new >= 0 and y_new < input.shape[0]):
                    # is it in the range of y
                    if (x_new >= 0 and x_new < input.shape[1]):
                        # is it in the range of x
                        mosaic_image[h][w] = input[y_new][x_new]
                        if(h < height_first):
                            height_first = h
                        if(w < width_first):
                            width_first = w
                        if(h-height_first<final_image.shape[0] and w - width_first < final_image.shape[1]):
                            final_image[h - height_first][w-width_first] = input[y_new][x_new]


    warpedIm = np.array(final_image, dtype='uint8')
    mergedIm = np.array(mosaic_image, dtype='uint8')
    #i'm not going to the right enogh

    return (warpedIm, mergedIm)