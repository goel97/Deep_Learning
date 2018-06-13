
# import the necessary packages
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

"""def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()
"""
def similar (imageA,imageB) :
	return ssim(imageA,imageB)

test = []

# load the images -- the original, the original + contrast,
# and the original + photoshop
test.append ( cv2.imread("images/test_01.jpg"))
test.append ( cv2.imread("images/test_02.jpg"))
test.append ( cv2.imread("images/test_03.jpg"))
train = cv2.imread("images/train_01.jpg")
#shopped = cv2.imread("images/jp_gates_photoshopped.png")

# convert the images to grayscale
test[0] = cv2.cvtColor(test[0], cv2.COLOR_BGR2GRAY)
test[1] = cv2.cvtColor(test[1], cv2.COLOR_BGR2GRAY)
test[2] = cv2.cvtColor(test[2], cv2.COLOR_BGR2GRAY)
train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

slot = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
slot[1][2] = -1

print (slot)
print ("\n")

# initialize the figure
#fig = plt.figure("Images")
#images = ("Original", original), ("Contrast", contrast)#, ("Photoshopped", shopped)

# loop over the images
'''for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")

# show the figure
plt.show()'''
n = 0
q = 0
X = [10 + (225*i) for i in range (0,5)]
Y = [4,425,821]
flag = input("Select Entry Gate : ")
flag1 = 0
aval = -1
for n in range (0,3):
	lot = test[n]
	slot = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
	slot[1][2] = -1
	m=0
	for y in Y:
		for  x in X:
			crop_img = lot[y:y+300, x:x+259]
			# compare the images
			#compare_images(crop_img, train, "Original vs. Original")
			s = similar(crop_img,train)
			t = mse (crop_img,train)
			if(t>1200):
				slot[(m)//5][(m)%5] = 1
			m = m+1
			print (str(m)+" : "+str(s) + "   " + str(t))
			#compare_images(original, shopped, "Original vs. Photoshopped")
	print("\n\n")	
	for i in range (0,3):
		print (slot[i])
	if(flag == 'X'):
		for p in range (0,5):
			if(slot[0][p] == 0):
				avail = 10 + (p + 1)
				flag1 = 1
				print ("\nClosest available parking slot : " + str(avail) + "\n")
				break
			elif(slot[1][p] == 0):
				avail = 20 + (p + 1)
				flag1 = 1
				print ("\nClosest available parking slot : " + str(avail) + "\n")
				break
		if (flag1 != 1):
			z = 2
			for i in range(0,3):
				if(slot[2][z-i] == 0):
					avail = 30 + (z-i+1)
					flag1 = 1
					print ("\nClosest available parking slot : " + str(avail) + "\n")
					break
				elif(slot[2][z+i] == 0):
					avail = 30 + (z-i+1)
					flag1 = 1
					print ("\nClosest available parking slot : " + str(avail) + "\n")
					break
		if(flag1 != 1):
			print("\nClosest available parking slot : Sorry! No Available Slots \n")
	else:
		for p in range (4,-1,-1):
			if(slot[2][p] == 0):
				avail = 30 + (p + 1)
				flag1 = 1
				print ("\nClosest available parking slot : " + str(avail) + "\n")
				break
			elif(slot[1][p] == 0):
				avail = 20 + (p + 1)
				flag1 = 1
				print ("\nClosest available parking slot : " + str(avail) + "\n")
				break
		if (flag1 != 1):
			z = 2
			for i in range(0,3):
				if(slot[0][z-i] == 0):
					avail = 10 + (z-i+1)
					flag1 = 1
					print ("\nClosest available parking slot : " + str(avail) + "\n")
					break
				elif(slot[0][z+i] == 0):
					avail = 10 + (z-i+1)
					flag1 = 1
					print ("\nClosest available parking slot : " + str(avail) + "\n")
					break
		if(flag1 != 1):
			print("\nClosest available parking slot : Sorry! No Available Slots \n")

	#print("\n\n")
