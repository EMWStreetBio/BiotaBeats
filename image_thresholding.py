'''Porting Wolfram image processing code to python'''
import cv2
import numpy as np
from scipy.spatial.distance import euclidean

'''Read and show image'''
# filename = "output_0027.png"
# filename = "BlobTest.jpg"
# filename = "yixiao.png"
# image = cv2.imread(filename)
# if image is None:
#     print "Image %s found" % (filename)
# else:
#     print "Processing %s" % (filename)
# cv2.imshow("Original", image)


def adaptiveThresholding(image, show=False):
    '''Blur image'''
    radius = 7
    kernel = cv2.getGaussianKernel(9, 3)
    gfImage = cv2.GaussianBlur(image,(radius,radius),0)
    # cv2.imshow("Blurred", gfImage)

    '''Binarize image'''
    gray = cv2.cvtColor(gfImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayScale", gray)
    # ret,binarized = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    binarized = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imshow("Binarized", binarized)

    ''' Morphological transformations (dilation, erosion, etc) '''
    kernel = np.ones((21,21),np.uint8)
    erosion = cv2.erode(binarized,kernel,iterations = 1)
    if show:
        cv2.imshow("Eroded", erosion)

    final = erosion

    # # yixiao.png does not show up with dilation
    # kernel = np.ones((5,5),np.uint8)
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)
    # cv2.imshow("Dilated", dilation)
    #
    # ret,final = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY_INV)

    '''Find centroids'''
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 255;

    # Filter by Area
    params.filterByArea = True
    params.minArea = 150

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.5

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(final)

    for point in keypoints:
        x = int(point.pt[0])
        y = int(point.pt[1])
        rad = point.size/2.0
        rad_int = int(rad)
        temp_window = gray[x-rad_int:x+rad_int, y-rad_int:y+rad_int]
        if (x>rad_int and x<image.shape[0]-rad_int and y>rad_int and y<image.shape[1]-rad_int):
            average_brightness = np.mean(temp_window)
        else:
            average_brightness = 0
        if show:
            print "keypoint: (%.3f, %.3f), radius %.3f, average brightness %d" % (point.pt[0], point.pt[1], rad, average_brightness)

    # Find out attributes of keypoints[0], point.pt, etc.
    # l = dir(keypoints[0].pt)
    # print  l
    # print keypoints[0].pt[0]

    '''Draw detected blobs as red circles'''
    if show:
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show centroids
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imwrite( "keypoints.jpg", im_with_keypoints)
        cv2.waitKey(0) # press any key while image is selected to escape
        cv2.destroyAllWindows()

    return keypoints
