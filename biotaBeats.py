import sys
import numpy as np
import os
import csv
if os.path.exists("PySynth-1.1/"):
    sys.path.append('PySynth-1.1')

# sys.path.append('/Users/gautam/.virtualenvs/cv/lib/python2.7/site-packages')
import cv2

import pysynth as ps
from midiutil.MidiFile import MIDIFile
from math import pi, floor
from scipy.spatial.distance import euclidean
from image_thresholding import *

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
    keypoints = detector.detect(img) #type: cv2.Keypoint

    # Show centroids
    if show:
        im_with_keypoints = cv2.drawKeypoints(orig, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)

    return keypoints

def analyze_notes(img, centroids, num_sectors=5):
    """ FINAL!!
        COL 0: note value
        COL 1: radial distance (pixels)
        COL 2: angle from top, center (radians)
        COL 3: centroid diameter (?)
        COL 4: average sector density (?) [0,1]
    """
    center = [(img.shape[0]/2.0), (img.shape[1]/2.0)]
    center_vector = [0.0, -(img.shape[1]/2.0)]
    sector_ang = (2.0*pi) / num_sectors
    note_info = np.zeros((len(centroids), 5)) # hard-coded!
    area_sums = np.zeros(num_sectors)
    for i in range(len(centroids)):
        note_info[i,1] = euclidean(centroids[i].pt, center) # fill ROW 1
        # may run into issues with parallel/antiparallel center_vector and pt_vector
        pt_vector = np.subtract(centroids[i].pt, center)
        num = np.dot(center_vector, pt_vector)
        denom = np.linalg.det([center_vector, pt_vector])
        angle = np.math.atan2(denom, num) #in range (-pi, pi]
        if (angle < 0.0): angle += (2.0*pi)
        note_info[i,2] = angle # fill ROW 2
        note_info[i,3] = centroids[i].size # fill ROW 3
        note_info[i,0] = (int(floor(angle / sector_ang))) # fill ROW 0
        area_sums[note_info[i,0]] += pi * ((note_info[i,3] / 2.0) ** 2)
    area_sums = area_sums / (0.5*sector_ang*(((img.shape[0]+img.shape[1])/2.0)**2))
    for i in range(len(centroids)):
        note_info[i,4] = area_sums[note_info[i,0]]
    return note_info

def save_csv(filename, note_info):
    with open(filename, 'wb') as csvfile:
        fwriter = csv.writer(csvfile)
        header = ["NOTE #", "SECTOR", "RAD DIST", "ANGLE (RAD)" , "DIAMETER", "SECTOR DENSITY"]
        fwriter.writerow(header)
        for i in range(len(note_info)):
            row = np.insert(note_info[i], 0, i+1)
            fwriter.writerow(row)
    csvfile.close()

def generate_music(img, note_info, algorithm, musicfile,
    octave_span=3, tot_time=60):
    '''
        img: Center of image should be center of circle. Cropped closely to size
        algorithm = 'concentric' or 'radial'
        note_info: 4xn numpy array with format [noteValue, radialDistance, angleFrom+YAxis, diameterOfColony]. n = number of notes
        musicfile: string specifying output filename
        octave_span: odd number of octaves to span notes across
        total_time = seconds
        possible future use: sorted_index = np.argsort()
    '''
    if (octave_span % 2 == 0) and (octave_span != 0):
        print "ERROR: 'octave_span' must be odd"
        return None
    if (octave_span > 9):
        print "ERROR: 'octave_span' beyond limits of MIDI" # technically exist octaves from 0-9.5
        return None
    if (type(octave_span) != int):
        print "Casting 'octave_span' %s to an integer %d" % (str(octave_span), int(octave_span))
        octave_span = int(octave_span)
    print musicfile

    # create your MIDI object
    mf = MIDIFile(5, adjust_origin=True) # only 1 track, changed adjust_origin to T
    track = 0   # the only track
    time = 0    # start at the beginning

    if (algorithm == 'concentric'): # where max = total_time
        sort_by = note_info[:,1] # radial distance
        max_time = (img.shape[0]+img.shape[1]) / 4.0 # max, radius (average)
    elif (algorithm == 'radial'): # normalization, where mean = 0; max - min = octave_span
        sort_by = note_info[:,2] # angle
        max_time = 2.0 * pi # aka 360 degrees
    else:
        print "ERROR: Invalid 'algorithm' mode"
        return None

    if (octave_span != 0): octave_factor = (np.amax(note_info[:,3])-np.amin(note_info[:,3])) / octave_span
    time_factor = tot_time / max_time
    mf.addTrackName(track, time, "Biota Beats Sample")
    mf.addTempo(track, time, 60) # tempo/bpm (param3) to be parametrized

    #print "sort_by", sort_by

    channel = 0
    note_conversion = {0:60, 1:62, 2:64, 3:67, 4:69} # hard-coded for pentatonic scale, may change

    # Generate notes
    print "Note\tPitch\trelOct\tStart(Beats)\tTrack"
    for i in range(note_info.shape[0]):
        if (octave_span != 0):
            octave = int(floor((note_info[i,3]-np.amin(note_info[:,3])) / octave_factor)) - (octave_span / 2)
        else: octave = 0
        pitch = note_conversion[note_info[i,0]] + (12 * octave) # 12 half-notes per octave
        time = sort_by[i] * time_factor
        '''BUG TODO: parameterize tempo to use total time.
        e.g. if we use 'angle' algorithm mode,
        and sort_by[i] = 2*pi radians, time_factor = 60/(2*pi)
        then final note is ALWAYS played at 60th beat'''
        duration = 5 # to be parameterized
        volume = 100 # to  be parameterized
        track = int(note_info[i,0])
        print i,"\t",pitch,"\t",octave,"\t",time,"\t",track
        mf.addNote(track, channel, pitch, time, duration, volume)

    # write it to disk
    with open(musicfile, 'wb') as outf:
        mf.writeFile(outf)

def main():
    # final, orig = img_processing("images/output_0027.png", show=False)
    #final, orig = img_processing("images/BlobTest.jpeg", inv=False, show=True)
    final, orig = img_processing("images/yixiao.png", dilate=0, inv=False)
    # centroids = find_centroids(final, orig, show=False)
    centroids = adaptiveThresholding(orig, show=False)
    #note_dist = rad_dist(final, centroids)
    #note_vals = sectorize(final, note_dist, 5)
    #radius = (final.shape[0]+final.shape[1])/4.0 # image currently not a perfect square
    #for i in range(len(note_vals)):
    #    print note_vals[i][0][0], note_vals[i][0][1], note_vals[i][1]
    #write_wav(note_vals, radius, "yixiao.wav")
    #write_midi(note_vals, radius, 60, "yixiao.mid")
    note_info = analyze_notes(final, centroids, 5)
    save_csv("output_data.csv", note_info)
    #generate_music(final, note_info, 'concentric', "music/yixiao_conc.mid", 3, 30)
    #generate_music(final, note_info, 'concentric', "music/yixiao_conc_no8.mid", 0, 30)
    #generate_music(final, note_info, 'radial', "music/yixiao_rad.mid", 3, 10) # doesn't make sense with sector notes
    #generate_music(final, note_info, 'radial', "music/yixiao_rad_no8.mid", 0, 10) # doesn't make sense with sector notes

if __name__=='__main__':
    # add arguments for image_location for testing... currently in main()
    if len(sys.argv)!=1:
        print 'USAGE: python biotaBeats.py'
    else:
        main()
