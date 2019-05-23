#!/usr/bin/env python
# coding: utf8

from matplotlib.mlab import find
from numpy import argmax, sqrt, mean, diff, log
from scipy.signal import blackmanharris, fftconvolve
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt4 import QtGui, QtCore
from matplotlib import figure
import sys
import threading
import atexit
import pyaudio
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


flag = 0
x = 0
idx = 0


class MicrophoneRecorder(object):
    def __init__(self, rate=44100, chunksize=3072):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def freq_from_autocorr(raw_data_signal, fs):
    corr = fftconvolve(raw_data_signal, raw_data_signal[::-1], mode='full')
    corr = corr[len(corr)/2:]
    d = diff(corr)
    start = find(d > 0)[0]
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px


def find_nearest(array, value):
    index = (np.abs(array - value)).argmin()
    return array[index]


def closest_value_index(array, guessValue):
    # Find closest element in the array, value wise
    closestValue = find_nearest(array, guessValue)
    # Find indices of closestValue
    indexArray = np.where(array == closestValue)
    # Numpys 'where' returns a 2D array with the element index as the value
    return indexArray[0][0]


def build_default_tuner_range():

    return {65.41: 'C2',
            69.30: 'C2#',
            73.42: 'D2',
            77.78: 'E2b',
            82.41: 'E2',
            87.31: 'F2',
            92.50: 'F2#',
            98.00: 'G2',
            103.80: 'G2#',
            110.00: 'A2',
            116.50: 'B2b',
            123.50: 'B2',
            130.80: 'C3',
            138.60: 'C3#',
            146.80: 'D3',
            155.60: 'E3b',
            164.80: 'E3',
            174.60: 'F3',
            185.00: 'F3#',
            196.00: 'G3',
            207.70: 'G3#',
            220.00: 'A3',
            233.10: 'B3b',
            246.90: 'B3',
            261.60: 'C4',
            277.20: 'C4#',
            293.70: 'D4',
            311.10: 'E4b',
            329.60: 'E4',
            349.20: 'F4',
            370.00: 'F4#',
            392.00: 'G4',
            415.30: 'G4#',
            440.00: 'A4',
            466.20: 'B4b',
            493.90: 'B4',
            523.30: 'C5',
            554.40: 'C5#',
            587.30: 'D5',
            622.30: 'E5b',
            659.30: 'E5',
            698.50: 'F5',
            740.00: 'F5#',
            784.00: 'G5',
            830.60: 'G5#',
            880.00: 'A5',
            932.30: 'B5b',
            987.80: 'B5',
            1047.00: 'C6',
            1109.0: 'C6#',
            1175.0: 'D6',
            1245.0: 'E6b',
            1319.0: 'E6',
            1397.0: 'F6',
            1480.0: 'F6#',
            1568.0: 'G6',
            1661.0: 'G6#',
            1760.0: 'A6',
            1865.0: 'B6b',
            1976.0: 'B6',
            2093.0: 'C7'
            }


class MplFigure(object):
    def __init__(self, parent):
        self.figure = figure.Figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)


class LiveFFTWidget(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # customize the UI
        self.initUI()

        # init class data
        self.initData()

        # connect slots
        self.connectSlots()

        # init MPL widget
        self.initMplWidget()

    def initUI(self):

        # primeira secao correspondente ao ganho e ao checkbox
        hbox_gain = QtGui.QHBoxLayout()
        autoGain = QtGui.QLabel('Auto gain for frequency spectrum')
        autoGainCheckBox = QtGui.QCheckBox(checked=False)
        hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(autoGainCheckBox)
        # reference to checkbox
        self.autoGainCheckBox = autoGainCheckBox

        # segunda secao correspondente ao ganho variavel
        hbox_fixedGain = QtGui.QHBoxLayout()
        fixedGain = QtGui.QLabel('Manual gain level for frequency spectrum')
        fixedGainSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        hbox_fixedGain.addWidget(fixedGain)
        hbox_fixedGain.addWidget(fixedGainSlider)
        # reference to slider
        self.fixedGainSlider = fixedGainSlider

        # nota musical
        hbox_nota = QtGui.QHBoxLayout()
        self.nota = QtGui.QLabel('Nota: La')
        #frequencia = QtGui.QLabel('Frequencia: 440Hz')
        hbox_nota.addWidget(self.nota)
        # hbox_nota.addWidget(frequencia)
        self.nota.setStyleSheet(
            "QLabel { color: rgb(50, 50, 50); font-size: 30px; background-color: rgba(188, 188, 188, 50);}")
        #frequencia.setStyleSheet("QLabel { color: rgb(50, 50, 50); font-size: 30px; background-color: rgba(188, 188, 188, 50);}")
        self.nota.setAlignment(QtCore.Qt.AlignCenter)
        # frequencia.setAlignment(QtCore.Qt.AlignCenter)

        # reference to slider

        vbox = QtGui.QVBoxLayout()

        # vbox.addLayout(hbox_gain)
        # vbox.addLayout(hbox_fixedGain)
        vbox.addLayout(hbox_nota)
        nota = self.setGeometry(QtCore.QRect(
            100, 100, 50, 50))  # (x, y, height, width)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)

        self.setLayout(vbox)

        self.setGeometry(500, 300, 350, 300)
        self.setWindowTitle('Reconhecedor de Notas Musicais')
        self.show()
        # timer for calls, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(50)
        # keep reference to timer
        self.timer = timer

    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()

        # keeps reference to mic
        self.mic = mic

        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize,
                                         1./mic.rate)
        #print mic.rate
        self.time_vect = np.arange(
            mic.chunksize, dtype=np.float32) / mic.rate * 1000

    def connectSlots(self):
        pass

    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(211)
        #self.ax_top.set_ylim(-32768, 32768)
        self.ax_top.set_ylim(-5000, 5000)
        self.ax_top.set_xlim(0, self.time_vect.max())
        #self.ax_top.set_xlim(0, 2500)
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)

        # bottom plot
        self.ax_bottom = self.main_figure.figure.add_subplot(212)
        self.ax_bottom.set_ylim(0, 2)
        self.ax_bottom.set_xlim(0, 3000)
        # self.freq_vect.max()
        self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)
        # line objects
        self.line_top, = self.ax_top.plot(self.time_vect,
                                          np.ones_like(self.time_vect))

        self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                                np.ones_like(self.freq_vect))

        # tight layout
        # plt.tight_layout()

    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """
        # gets the latest frames
        frames = self.mic.get_frames()

        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # plots the time signal
            self.line_top.set_data(self.time_vect, current_frame)
            # computes and plots the fft signal
            fft_frame = np.fft.rfft(current_frame)

            inputnote = round(freq_from_autocorr(current_frame, 44100), 4)

            tunerNotes = build_default_tuner_range()
            frequencies = np.array(sorted(tunerNotes.keys()))
            targetnote = closest_value_index(frequencies, round(inputnote, 2))

            if inputnote < 3000:
                print "INPUTNOTE", inputnote, str(
                    tunerNotes[frequencies[targetnote]])
                self.nota.setText(str(tunerNotes[frequencies[targetnote]]))
            else:
                self.nota.setText("")

            if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                fft_frame /= np.abs(fft_frame).max()
            else:
                fft_frame *= (1 + 20) / 5000000.

            # self.fixedGainSlider.value()

            freqs = np.fft.fftfreq(len(fft_frame))
            idx = np.argmax(np.abs(fft_frame))

            x, y = parabolic(fft_frame, idx)
            #print "parabolic",10000/x
            #print "freq",freqs[10000/x]

            freq = freqs[idx]

            self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))

            global flag
            if fft_frame[np.argmax(fft_frame)] > 1.1:

                if flag == 0:
                    global x
                    for x in range(0, len(fft_frame)):
                        global x
                        if np.abs(fft_frame[x]) > 0.5:
                            global idx
                            idx = x
                            global flag
                            flag = 1
                            break

                if fft_frame[np.argmax(fft_frame)] < 0.5:
                    global flag
                    flag = 0

                if flag == 1:
                    frequencia = self.freq_vect[idx]

                    #print frequencia,np.abs(fft_frame[x])

                    #razoes = np.array([1,1.06666,1.125,1.2,1.25,1.333333,np.sqrt(2),1.5,1.6,1.666666,1.75,1.875,0])

                    #print abs(razoes-(inputnote/65.2090221154)%2)

                    #print "nota",np.argmin(abs(razoes-(inputnote/65.2090221154)%2))
                    #print "n",(inputnote/65)%2
                    #print (frequencia/63.4765625)%2

            #print self.freq_vect[idx],np.abs(fft_frame[x])

            # refreshes the plots
            self.main_figure.canvas.draw()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = LiveFFTWidget()
    sys.exit(app.exec_())
