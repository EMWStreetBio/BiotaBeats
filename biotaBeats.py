import sys
#import time
import numpy as np

import sys
import os
if os.path.exists("PySynth-1.1/"):
    sys.path.append('PySynth-1.1')

# sys.path.append('/Users/gautam/.virtualenvs/cv/lib/python2.7/site-packages')
import cv2

import pysynth as ps
from midiutil.MidiFile import MIDIFile
from math import pi, floor
from scipy.spatial.distance import euclidean
import image_thresholding

def img_processing(imgfile, erode=1, dilate=1, inv=True, show=False):
    ''' STEP-BY-STEP:
       1) Gaussian filtering;
       2) image grayscale + binarization;
       3) image erosion (OPTIONAL)
       4) image dilation;
    '''
    orig = cv2.imread(imgfile)
    if show: cv2.imshow("Original", orig)

    radius = 7
    kernel = cv2.getGaussianKernel(9, 3)
    image = cv2.GaussianBlur(orig,(radius,radius),0)
    if show: cv2.imshow("Blurred", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show: cv2.imshow("GrayScale", image)
    ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY) # unnessary? should just be INV?
    if show: cv2.imshow("Binarized", image)

    kernel = np.ones((15,15),np.uint8)
    if (erode != 0):
        image = cv2.erode(image,kernel,iterations = erode)
        if show: cv2.imshow("Eroded", image)
    if (dilate != 0):
        image = cv2.dilate(image,kernel,iterations = dilate)
        if show: cv2.imshow("Dilated", image)
    if inv: ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)

    if show: cv2.waitKey(0)
    #final = erImage
    return image, orig

def find_centroids(img, orig, show=False): # finds dark spots only
    # centroidData = find centroids
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255
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
    keypoints = detector.detect(img) #type: cv.Keypoint

    # Show centroids
    if show:
        im_with_keypoints = cv2.drawKeypoints(orig, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)

    return keypoints

def rad_dist(img, centroids):
    center = [(img.shape[0]/2.0), (img.shape[1]/2.0)]
    rad_dist = np.zeros(len(centroids))
    for i in range(len(centroids)):
        rad_dist[i] = euclidean(centroids[i].pt, center)
    notes = zip(rad_dist, centroids)
    notes.sort()
    return notes

def sectorize(img, notes, num_sectors):
    sector_ang = (2.0*pi) / num_sectors
    center = [(img.shape[0]/2.0), (img.shape[1]/2.0)]
    center_vector = [0.0, -(img.shape[1]/2.0)]
    note_vals = []
    for i in range(len(notes)):
        pt_vector = np.subtract(notes[i][1].pt, center)
        # may run into issues with parallel/antiparallel center_vector and pt_vector
        num = np.dot(center_vector, pt_vector)
        denom = np.linalg.det([center_vector, pt_vector])
        angle = np.math.atan2(denom, num)
        if (angle < 0.0): angle += (2.0*pi)
        note_vals.append(int(floor(angle / sector_ang)))
    return zip(notes, note_vals) # FIX: currently zips ((a,b)c)

def write_wav(note_vals, total_dist, musicfile): # add total_time?
    # most basic, plays all notes w/o regard for rests or relative timing
    # hard-coded for pentatonic scale:
    note_conversion = {0:'c', 1:'d', 2:'e', 3:'g', 4:'a'}
    rests = np.zeros(len(note_vals)) # may be unnecessary
    rests[0] = note_vals[0][0][0]
    for i in range(1,len(note_vals)):
        rests[i] = note_vals[i][0][0] - note_vals[i-1][0][0]
    # simply plays all notes as quarter notes (no rests)
    # PySynth cannot play multiple notes at the same time, which would be preferable
    notes = []
    for i in note_vals:
        notes.append((note_conversion[i[1]], 4))
    notes = tuple(notes)
    ps.make_wav(notes, fn = musicfile)

def write_midi(note_vals, total_dist, total_time, musicfile):
    # create your MIDI object
    mf = MIDIFile(1, adjust_origin=True)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    time_factor = total_time / total_dist
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, (1/time_factor))

    channel = 0
    note_conversion = {0:60, 1:62, 2:64, 3:67, 4:69}

    for i in note_vals: # add some notes
        pitch = note_conversion[i[1]]
        time = i[0][0] * time_factor
        duration = 4 # can be parameterized
        volume = 100 # can be parameterized
        mf.addNote(track, channel, pitch, time, duration, volume)

    ''' http://stackoverflow.com/questions/11059801/how-can-i-write-a-midi-file-with-python
    how to write a note (each pitch # is a piano key)
    pitch = 60           # C4 (middle C)
    time = 0             # start on beat 0
    duration = 1         # 1 beat long
    mf.addNote(track, channel, pitch, time, duration, volume)
    '''

    # write it to disk
    with open(musicfile, 'wb') as outf:
        mf.writeFile(outf)

def main():
    # final, orig = img_processing("images/output_0027.png", show=False)
    #final, orig = img_processing("images/BlobTest.jpeg", inv=False, show=True)
    final, orig = img_processing("images/yixiao.png", dilate=0, inv=False)
    # centroids = find_centroids(final, orig)
    centroids = image_thresholding.adaptiveThresholding(orig)
    note_dist = rad_dist(final, centroids)
    note_vals = sectorize(final, note_dist, 5)
    radius = (final.shape[0]+final.shape[1])/4.0 # image currently not a perfect square
    for i in range(len(note_vals)):
        print note_vals[i][0][0], note_vals[i][0][1], note_vals[i][1]
    # write_wav(note_vals, radius, "yixiao.wav")
    # write_midi(note_vals, radius, 5, "yixiao.mid")

if __name__=='__main__':
    # add arguments for image_location for testing... currently in main()
    if len(sys.argv)!=1:
        print 'USAGE: python biotaBeats.py'
    else:
        main()
