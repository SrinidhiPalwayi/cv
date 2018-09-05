import matplotlib.pylab as plt
import numpy as np

""""
run script twice
change the list to the other image and another text file
"""

"""""
right_file = "right.png"
left_file = "left.png"
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

def computeH():
    orig_img_size = 2160
    new_img_size = 2.0
    scaler = lambda t: t*(new_img_size/float(orig_img_size))
    f = open("left.txt")
    left_list = np.asarray([json.loads(line.strip()) for line in f])
    left_list = np.asarray([[scaler(i) for i in coord] for coord in left_list])
    print(left_list)
    f.close()

    f = open("left.txt")
    right_list = np.asarray([json.loads(line.strip()) for line in f])
    right_list = np.asarray([[scaler(i) for i in coord] for coord in right_list])
    f.close()

    n = len(left_list)

    left_x = np.asarray([item[0] for item in left_list])
    left_y = np.asarray([item[1] for item in right_list])

    right_x = np.asarray([item[0] for item in right_list])
    transpose_right_x = np.reshape(right_x, (n,1))
    right_y = np.asarray([item[1] for item in right_list])
    transpose_right_y = np.reshape(right_y, (n,1))

    one_array = np.ones((n,1), dtype=int)
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

#one_list= np.ones((1,n), dtype=int)
#left_side = np.concatenate((np.transpose(left_list), one_list), axis=0)

#transformed = np.dot(H, left_side)

#print (vh.shape)
#print(A.shape)
#print(H)

#print(right_list)
print(computeH())
