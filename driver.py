from computeH import computeH
from warpImage import warpImage
import numpy as np
import matplotlib.pyplot as plt

rectangle = np.asarray([[0,134,0,134],[0,0,101,101]])
np.save("rect.npy", rectangle)
print(rectangle.shape)

#H = computeH('picassopoints.npy', "rect.npy",)
H = computeH('left.npy', "right.npy",)
warpImage, mergedImage = warpImage("left.jpg","right.jpg", H)

plt.imshow(mergedImage)
plt.savefig('stereo1.png')
plt.show()
