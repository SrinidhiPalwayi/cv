from computeH import computeH
from warpImage import warpImage
import numpy as np
import matplotlib.pyplot as plt
import PIL

fig,axis = plt.subplots(4,2, figsize =(12,12))
fig.suptitle('Images')
images =[]

Hcrop = computeH("cc1.npy", "cc2.npy")
warpImageCrop, mergedImageCrop = warpImage("crop1.jpg", "crop2.jpg", Hcrop)
images.append(axis[0,0].imshow(warpImageCrop))
axis[0,0].set_title('Warped Crop Circle')


images.append(axis[0,1].imshow(mergedImageCrop))
axis[0,1].set_title('Merged Crop Circle')

Hbuilding = computeH("wdc1.npy", "wdc2.npy")
warpImageBuilding, mergedImageBuilding = warpImage("wdc1.jpg", "wdc2.jpg", Hbuilding)
images.append(axis[1,0].imshow(warpImageBuilding))
axis[1,0].set_title('Warped WDC')

images.append(axis[1,1].imshow(mergedImageBuilding))
axis[1,1].set_title('Merged WDC')

Hmine = computeH("left.npy", "right.npy")
warpImageMine, mergedImageMine = warpImage("left.jpg", "right.jpg", Hmine)
images.append(axis[2,0].imshow(warpImageMine))
axis[2,0].set_title('Warped My Graphic')

images.append(axis[2,1].imshow(mergedImageMine))
axis[2,1].set_title('Merged My Graphic')

Hpic = computeH("rect.npy", "picasso.npy")
warpImageMine, mergedImageMine = warpImage("scream.jpg", "picasso.jpg", Hpic)
images.append(axis[3,0].imshow(warpImageMine))
axis[3,0].set_title('Warped On to Frame')

images.append(axis[3,1].imshow(mergedImageMine))
axis[3,1].set_title('Merged On to Frame')
""""
Hpicasso = computeH("rect.npy", "picasso.npy")
warpImageBuilding, mergedImageBuilding = warpImage("wdc1.jpg", "wdc2.jpg", H)
images.append(axis[1,0].imshow(warpImageBuilding, aspect ='auto'))
axis[1,0].set_title('Warped WDC')
"""





#H = computeH("left.npy",'right.npy')
#H = computeH('left.npy', "right.npy")
#warpImage, mergedImage = warpImage("right.jpg", "left.jpg", H)
#warpImage, mergedImage = warpImage( "left.jpg", "right.jpg", H)

#plt.imshow(warpImage)
#plt.savefig('picasso_splatter.png')
plt.show()
