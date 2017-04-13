import sys
import os
if os.path.exists("PySynth-1.1/"):
    sys.path.append('PySynth-1.1')
import pysynth

test = ( ('c', 4), ('d', 4), ('e', 4), ('f', 4) )
pysynth.make_wav(test, fn = "test.wav")

song = (('c', 4), ('c*', 4), ('e', 4), ('g', 4), ('g*', 2), ('g5', 4), ('g5*', 4), ('r', 4), ('e5', 4), ('e5*', 4))
pysynth.make_wav(song, fn = "song.mid")
