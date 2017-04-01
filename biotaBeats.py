import cv2
import sys
import numpy as np
import pysynth as ps
from scipy.spatial.distance import euclidean

def img_processing(imgfile):
    # can be broken down later
    ''' STEP-BY-STEP:
       1) Gaussian filtering; 2) image binarization;
       3) image erosion; 4) image dilation;
    '''
    image = cv2.imread(imgfile)
    cv2.imshow("Original", image)

    # gfImage = blur image (rad = 3)
    radius = 7
    kernel = cv2.getGaussianKernel(9, 3)
    gfImage = cv2.GaussianBlur(image,(radius,radius),0)
    cv2.imshow("Blurred", gfImage)

    # binarized = binarize
    grayImage = cv2.cvtColor(gfImage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GrayScale", grayImage)
    ret,binImage = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
    cv2.imshow("Binarized", binImage)

    # erosion
    kernel = np.ones((15,15),np.uint8)
    erImage = cv2.erode(binImage,kernel,iterations = 1)
    cv2.imshow("Eroded", erImage)
    '''CURRENTLY COMMENTED OUT WHILE TESTING YIXIAO.PNG
    # dilation
    diImage = cv2.dilate(erImage,kernel,iterations = 1)
    ret,dilationInv = cv2.threshold(diImage,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Dilated", diImage)
    final = diImage
    '''
    final = erImage
    return final, image

def find_centroids(img, orig):
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
    keypoints = detector.detect(img)

    # Show centroids
    im_with_keypoints = cv2.drawKeypoints(orig, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)

    return keypoints

def rad_dist(img, centroids):
    center = ((img.shape[0]/2.0), (img.shape[1]/2.0))
    top_center = ((img.shape[0]/2.0), 0.0)
    rad_dist = np.zeros(len(centroids))
    for i in range(len(centroids)):
        rad_dist[i] = euclidean(centroids[i].pt, center)
    notes = zip(rad_dist, centroids)
    notes.sort()
    return notes

def main():
    # image = cv2.imread("jurassic_world.jpg")
    # image = cv2.imread("output_0027.png")
    # image = cv2.imread("BlobTest.jpg")
    final, orig = img_processing("yixiao.png")
    centroids = find_centroids(final, orig)
    for keypoint in centroids:
        print "centroid: " + str(keypoint.pt)

    note_dist = rad_dist(final, centroids)
    for i in range(len(note_dist)):
        print note_dist[i][0], note_dist[i][1].pt

    cv2.waitKey(0) # not sure where this should go

if __name__=='__main__':
    # add arguments for image_location for testing... currently in main()
    if len(sys.argv)!=1:
        print 'Usage: python biotaBeats.py'
    else:
        main()
