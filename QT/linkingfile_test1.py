
import numpy as np
import test1

import sys
import matplotlib


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import qdarkstyle

matplotlib.use('Qt5Agg')
import numpy as np
import scipy.signal

#TO GENERATE DESIGN FILE: pyuic5 test1.ui -o test1.py

class MRSignal():

    def __init__(self):
        self.f0 = 0
        self.IF = 0
        self.larmor = 0
        self.larmor_end = 0
        self.dt = 0
        self.BW = 0
        self.Nsignals = 0
        self.T = 0
        self.t = np.arange(0)
        self.RFsignal = np.zeros(0)
        self.addNoise = False
        self.noiseLvl = 10
        self.T2 = 50e-6
        self.rollN = 0
        self.t0 = 0
        self.t1 = 0
#         self.generateRFSignal()
        
    def generateRFSignal(self,Bfield):
        
        self.t = np.arange(self.t0,self.t1,self.dt)
        
        self.RFsignal = np.zeros(len(self.t))
        for i in range(0,self.Nsignals):
            f = np.linspace(self.larmor,self.larmor_end,len(self.t)) - self.BW/2 + self.BW/2 * i / self.Nsignals
            self.RFsignal = self.RFsignal + np.sin(2*np.pi*np.multiply(f,self.t))
        self.RFsignal = np.roll(self.RFsignal,self.rollN)
        self.RFsignal = self.RFsignal * np.exp(-self.t/self.T2)
        if self.addNoise:
            self.RFsignal = self.RFsignal + np.random.normal(0, self.noiseLvl, len(self.t))
        self.LO_I = np.sin(2*np.pi*self.f0*self.t)
        self.LO_Q = np.sin(2*np.pi*self.f0*self.t + np.pi/2)
        self.I = np.multiply(self.RFsignal,self.LO_I)
        self.Q = np.multiply(self.RFsignal,self.LO_Q)
        
    def calculatePhase(self):
        self.phase = np.unwrap(2*np.arctan(self.Q/self.I))/2
        
        
class Field():
    def __init__(self):
        self.B0 = 3 #Tesla
        self.xGradientWaveform = np.load(r"C:\Users\Oskar\Dropbox\Local files_oskars dator\Dropbox dokument\Python Scripts\General MRI\data\XGradientWaveform.npy")
        self.Gx_max = 0.05 #Tesla/m
        self.snippetStart = 10/1e3 #ms
        self.snippetLength = 1/1e3 #ms
        
    def setGrid(self,x0,x1,y0,y1):
        res = (x1-x0)/1000 #Resolution
        self.xPts = np.arange(x0,x1,res)
        self.yPts = np.arange(y0,y1,res)
        self.xGrid, self.yGrid = np.meshgrid(self.xPts,self.yPts, sparse=True)
        
    def setBField(self):
        
        Bgrid = self.B0 * np.ones(np.shape(self.xGrid))
        
        
        

class Window(QtWidgets.QMainWindow, test1.Ui_Dialog):
    def __init__(self):
        
        self.Bfield = Field()
        self.signal = MRSignal()
        self.signal2 = MRSignal()
        super(self.__class__, self).__init__()
        self.setupUi(self)  # This is defined in design.py file automatically
        
        self.setWindowTitle("MR signals")
        
        self.initLineEdit()
        
        self.pushButton.clicked.connect(self.updateParameters)
        self.pushButton_2.clicked.connect(self.togglePages)
        self.pushButton_3.clicked.connect(self.togglePages)
#         self.checkBox.stateChanged.connect(self.enableGradientRamp)
        self.region = pg.LinearRegionItem(values=(0.001,0.001001))
        self.region2 = pg.LinearRegionItem(values=(0.005,0.005001),brush=(255,0,0,50))
#         self.region.sigRegionChanged.connect(self.setRegionTextBox)
#         self.region2.sigRegionChanged.connect(self.setRegionTextBox)
        self.lineEdit_33.textChanged.connect(self.setRegionGraphics)
        self.lineEdit_34.textChanged.connect(self.setRegionGraphics)
        self.lineEdit_35.textChanged.connect(self.setRegionGraphics)
        self.lineEdit_36.textChanged.connect(self.setRegionGraphics)
        self.setRegionGraphics()
    
    def setRegionGraphics(self):
        self.region.setRegion([np.double(self.lineEdit_33.text()),np.double(self.lineEdit_34.text())])
        self.region2.setRegion([np.double(self.lineEdit_36.text()),np.double(self.lineEdit_35.text())])

        
#     def setRegionTextBox(self):
#         self.lineEdit_33.setText(str(np.around(self.region.getRegion()[0],8)))
#         self.lineEdit_34.setText(str(np.around(self.region.getRegion()[1],8)))
#         self.lineEdit_36.setText(str(np.around(self.region2.getRegion()[0],8)))
#         self.lineEdit_35.setText(str(np.around(self.region2.getRegion()[1],8)))

        
    def togglePages(self):
        if self.pushButton_2.isEnabled():
            self.stackedWidget.setCurrentIndex(0)
            self.pushButton_2.setEnabled(False)
            self.pushButton_3.setEnabled(True)
        elif self.pushButton_3.isEnabled():
            self.stackedWidget.setCurrentIndex(1)
            self.pushButton_3.setEnabled(False)
            self.pushButton_2.setEnabled(True)
        
        
        
#     def enableGradientRamp(self):
#         if self.checkBox.isChecked():
#             self.lineEdit_10.setEnabled(True)
#             self.lineEdit_11.setEnabled(True)
#             self.signal.larmor_end = 12e6
#             self.signal2.larmor_end = 12e6
#             self.lineEdit_10.setText('127.8e6')
#             self.lineEdit_11.setText('128.26e6')
#         else:
#             self.lineEdit_10.setEnabled(False)
#             self.lineEdit_11.setEnabled(False)   
            
#     def setLarmorEndFreq(self):
#         if self.checkBox.isChecked():
#             self.signal.larmor_end = np.double(self.lineEdit_10.text())
#             self.signal2.larmor_end = np.double(self.lineEdit_11.text())
#         else:
#             self.signal.larmor_end = self.signal.larmor
#             self.signal2.larmor_end = self.signal2.larmor
        
    def zeroPad(self,array):
        zeros = np.zeros(self.Nbox)
        array = np.concatenate((array,zeros))
        return array
        
    def LPfilter(self):
        if self.checkBox_2.isChecked():
            self.Nbox = np.int(self.lineEdit_15.text())
            self.signal.I = self.zeroPad(self.signal.I)
            self.signal.Q = self.zeroPad(self.signal.Q)
            self.signal2.I = self.zeroPad(self.signal2.I)
            self.signal2.Q = self.zeroPad(self.signal2.Q)
            self.signal.I = np.convolve(np.ones(self.Nbox)/self.Nbox,self.signal.I,mode='valid')[np.int(self.Nbox/2):np.int(len(self.signal.t)+self.Nbox/2)]
            self.signal.Q = np.convolve(np.ones(self.Nbox)/self.Nbox,self.signal.Q,mode='valid')[np.int(self.Nbox/2):np.int(len(self.signal.t)+self.Nbox/2)]
            self.signal2.I = np.convolve(np.ones(self.Nbox)/self.Nbox,self.signal2.I,mode='valid')[np.int(self.Nbox/2):np.int(len(self.signal2.t)+self.Nbox/2)]
            self.signal2.Q = np.convolve(np.ones(self.Nbox)/self.Nbox,self.signal2.Q,mode='valid')[np.int(self.Nbox/2):np.int(len(self.signal2.t)+self.Nbox/2)]

    def updateParameters(self):
        self.signal.larmor = np.double(self.lineEdit_3.text())
        self.signal2.larmor = np.double(self.lineEdit_7.text())
        self.signal.addNoise = self.checkBox_9.isChecked()
        self.signal.noiseLvl = np.double(self.lineEdit_31.text())
        self.signal.T2 = np.double(self.lineEdit_32.text())
        self.signal.rollN = np.int(self.lineEdit_13.text())
        self.signal2.addNoise = self.checkBox_9.isChecked()
        self.signal2.noiseLvl = np.double(self.lineEdit_31.text())
        self.signal2.T2 = np.double(self.lineEdit_32.text())
        self.signal2.rollN = np.int(self.lineEdit_13.text())
        self.signal.f0 = np.double(self.lineEdit.text())
        self.signal.dt = np.double(self.lineEdit_4.text())
        self.signal2.dt = np.double(self.lineEdit_4.text())
        self.signal.BW = np.double(self.lineEdit_5.text())
        self.signal.Nsignals = np.int(self.lineEdit_6.text())
        self.signal.T = np.double(self.lineEdit_2.text())
        self.signal.t0 = np.double(self.lineEdit_33.text())/1e3
        self.signal2.t0 = np.double(self.lineEdit_36.text())/1e3
        self.signal.t1 = np.double(self.lineEdit_34.text())/1e3
        self.signal2.t1 = np.double(self.lineEdit_35.text())/1e3
        self.signal2.f0 = np.double(self.lineEdit.text())
        self.signal2.BW = np.double(self.lineEdit_8.text())
        self.signal2.Nsignals = np.int(self.lineEdit_9.text())
        self.signal2.T = np.double(self.lineEdit_2.text())
        self.signal.generateRFSignal(self.Bfield)
        self.signal2.generateRFSignal(self.Bfield)
        self.LPfilter()
        self.signal.calculatePhase()
        self.signal2.calculatePhase()
        self.plot()
#         print('RF max: ' + str(np.max(self.signal.RFsignal)))
#         print ("f0: " + str(self.signal.f0))
#         print ("IF: " + str(self.signal.IF))
#         print ("Larmor: " + str(self.signal.larmor))
#         print ("dt: " + str(self.signal.dt))
#         print ("BW: " + str(self.signal.BW))
#         print ("Nsignals: " + str(self.signal.Nsignals))

    def plot(self):
        self.graphicsView.clear()
        self.graphicsView_2.clear()
        self.graphicsView_3.clear()
        self.graphicsView_4.clear()
        
        self.graphicsView.plot(self.signal.t, self.signal.RFsignal,pen=pg.mkPen('r', width=1))
        self.graphicsView.plot(self.signal2.t, self.signal2.RFsignal,pen=pg.mkPen('g', width=1))
        if self.checkBox_6.isChecked():
            self.graphicsView_2.plot(self.signal.t, self.signal.I,pen=pg.mkPen('r', width=1))
        if self.checkBox_8.isChecked():
            self.graphicsView_2.plot(self.signal2.t, self.signal2.I,pen=pg.mkPen('g', width=1))
        if self.checkBox_5.isChecked():
            self.graphicsView_2.plot(self.signal.t, self.signal.Q,pen=pg.mkPen('b', width=1))
        if self.checkBox_7.isChecked():
            self.graphicsView_2.plot(self.signal2.t, self.signal2.Q,pen=pg.mkPen('y', width=1))

        self.graphicsView_3.plot(self.signal.I,self.signal.Q,pen=pg.mkPen('r', width=1))
        self.graphicsView_3.plot(self.signal2.I,self.signal2.Q,pen=pg.mkPen('g', width=1))
        
        self.graphicsView_4.plot(self.signal.t,self.signal.phase,pen=pg.mkPen('r', width=1))
        self.graphicsView_4.plot(self.signal2.t,self.signal2.phase,pen=pg.mkPen('g', width=1))
        
        self.graphicsView_5.plot(self.Bfield.xGradientWaveform[0,:],self.Bfield.xGradientWaveform[1,:])
#         self.graphicsView_5.LinearRegionItem(values=(0, 0.1), orientation='vertical', brush=None, pen=None, hoverBrush=None, hoverPen=None, movable=True, bounds=(0,0.1), span=(0, 1), swapMode='sort')
        self.graphicsView_5.addItem(self.region, ignoreBounds=True)
        self.graphicsView_5.addItem(self.region2, ignoreBounds=True)
      
    def initLineEdit(self):
        self.lineEdit.setText('126e6') #LO freq
        self.lineEdit_3.setText('127.74e6')
        self.lineEdit_4.setText('0.1e-9')
        self.lineEdit_5.setText('100e3')
        self.lineEdit_6.setText('100')
        self.lineEdit_13.setText('0')
#         self.lineEdit_11.setText('10e6')
        self.lineEdit_7.setText('128e6')
        self.lineEdit_8.setText('100e3')
        self.lineEdit_9.setText('100')
        self.lineEdit_14.setText('0')
        self.lineEdit_33.setText('0.001')
        self.lineEdit_34.setText('0.001001')
        self.lineEdit_36.setText('0.005')
        self.lineEdit_35.setText('0.005001')
def main():
    app = QtWidgets.QApplication(sys.argv)  # A new instance of QApplication
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside = False))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    form = Window()                 # We set the form to be our ExampleApp (design)
    form.show()                         # Show the form
    app.exec_()                         # and execute the app


if __name__ == '__main__':              # if we're running file directly and not importing it
    main()  
    
