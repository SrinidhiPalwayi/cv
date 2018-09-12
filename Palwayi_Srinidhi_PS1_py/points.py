import matplotlib.pylab as plt
import numpy as np

"change the image names, and .npy files every time you run the program"

left_file = "left.jpg"
right_file = "right.jpg"

r_img = plt.imread(right_file)
l_img = plt.imread(left_file)
f,a = plt.subplots(1,2, figsize=(10,10))
a[0].imshow(l_img)
a[1].imshow(r_img)
pos_right = []
pos_left = []
click = 0
def onclick(event):
    global click
    print "came in "
    if(click%2==0):
        pos_left.append([event.xdata,event.ydata])
    else:
        pos_right.append([event.xdata,event.ydata])
    click+=1

f.canvas.mpl_connect('button_press_event', onclick)
f.show()
plt.show()
zero = np.zeros((2,2,4))
print(pos_left)
print(pos_right)

pos_left = np.asarray(pos_left)
np.save("left.npy", pos_left)
pos_left = np.transpose(pos_left)

pos_right = np.asarray(pos_right)
np.save("right.npy", pos_right)

pos_right = np.transpose(pos_right)

pos_right = np.asarray([pos_right])
pos_left = np.asarray([pos_left])

together = np.vstack((pos_left, pos_right))


print(together.shape)

#need to remake points
#np.save("points.npy", together)
print(together)

