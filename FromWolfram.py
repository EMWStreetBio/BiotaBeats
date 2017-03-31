import cv2
import numpy as np

from scipy.spatial.distance import euclidean

# images = read image
# image = cv2.imread("jurassic_world.jpg")
# image = cv2.imread("output_0027.png")
image = cv2.imread("yixiao.png")
# image = cv2.imread("BlobTest.jpg")
cv2.imshow("Original", image)

# gfImage = blur image (rad = 3)
radius = 7
kernel = cv2.getGaussianKernel(9, 3)
gfImage = cv2.GaussianBlur(image,(radius,radius),0)
cv2.imshow("Blurred", gfImage)

# binarized = binarize
gray = cv2.cvtColor(gfImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale", gray)
ret,binarized = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binarized", binarized)

kernel = np.ones((15,15),np.uint8)
erosion = cv2.erode(binarized,kernel,iterations = 1)
cv2.imshow("Eroded", erosion)

# yixiao does not show up with dilation
'''
dilation = cv2.dilate(erosion,kernel,iterations = 1)
cv2.imshow("Dilated", dilation)

ret,dilationInv = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY_INV)

final = dilationInv
'''
final = erosion

# centroidData = find centroids
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 150

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(final)

for point in keypoints:
    print "keypoint: " + str(point.pt)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show centroids
cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0) # press any key while image is selected to escape

center = ((image.shape[0]/2.0), (image.shape[1]/2.0))
top_center = ((image.shape[0]/2.0), 0.0)
rad_dist = np.zeros(len(keypoints))
for i in range(len(keypoints)):
    rad_dist[i] = euclidean(keypoints[i].pt, center)
staff = zip(rad_dist, keypoints)
staff.sort()

for i in range(len(staff)):
    print staff[i][0], staff[i][1].pt
