
import numpy as np
import test1

import sys
import matplotlib
from scipy.interpolate import interp1d

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import qdarkstyle
from scipy import interpolate

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
        self.I = 0
        self.Q = 0
        self.LO_I = 0
        self.LO_Q = 0
        self.addNoise = True
        self.noiseLvl = 0.1
        self.T2 = 50e-6
        self.rollN = 0
        self.t0 = 0
        self.t1 = 0
        self.gyroMagneticRatio = 127.74e6/3 #Hz/T
        self.doLowPass = True
        self.Probe = Probe()
        self.RGB = 1000
#         self.generateRFSignal()
        
    def generateRFSignal(self,Bfield,Probe):
        
        self.t = np.arange(0,(self.t1-self.t0),self.dt) #Always starts from zero for each new excitation!
        Gx_temp=interp1d(Bfield.xGradientWaveform[0,:],Bfield.xGradientWaveform[1,:])
        Gy_temp=interp1d(Bfield.yGradientWaveform[0,:],Bfield.yGradientWaveform[1,:])
        xPos_temp=interp1d(Probe.xPos[0,:],Probe.xPos[1,:])
        yPos_temp=interp1d(Probe.yPos[0,:],Probe.yPos[1,:])
        Gx_interp = Gx_temp(self.t+self.t0)
        Gy_interp = Gy_temp(self.t+self.t0)
        xPos_interp = xPos_temp(self.t+self.t0)
        yPos_interp = yPos_temp(self.t+self.t0)
        f = self.gyroMagneticRatio * (Bfield.B0 + (Gx_interp*Bfield.Gx_max*xPos_interp + Gy_interp*Bfield.Gy_max*yPos_interp))
#         f = (1/(2*np.pi)) * self.gyroMagneticRatio * Bfield.B0
        self.RFsignal = np.zeros(len(self.t))
        self.RFsignal = np.sin(2*np.pi*np.multiply(f,self.t)) #Oscillation
        self.RFsignal = self.RFsignal * np.exp(-self.t/self.T2) #Decay
        if self.addNoise:
            self.RFsignal = self.RFsignal + np.random.normal(0, self.noiseLvl, len(self.t))
        self.LO_I = np.sin(2*np.pi*self.f0*self.t)
        self.LO_Q = np.sin(2*np.pi*self.f0*self.t + np.pi/2)
        self.I = np.multiply(self.RFsignal,self.LO_I)
        self.Q = np.multiply(self.RFsignal,self.LO_Q)
        self.calculateFFTs()
        
    def calculatePhase(self):
#         self.phase = np.unwrap(2*np.arctan(self.Q/self.I))/2
        self.phase = self.unWrapPhaseSimple(np.arctan(self.Q/self.I))
        self.differentiatePhase()
        
    def differentiatePhase(self):
        self.diffPhase=np.diff(self.phase)/self.dt
        A = np.vstack([self.t**2, self.t, np.ones(len(self.t))]).T
        self.c2, self.c1, self.c0 = np.linalg.lstsq(A, self.phase, rcond=None)[0]
        
    def calculateFFTs(self):
        self.fft_freq = np.linspace(0,0.5/(self.dt),np.int(len(self.I)/2))
        self.fft = np.fft.fft(self.I)
    
    def unWrapPhaseSimple(self,p):
        discont = np.pi/2       #Set discontinuity
        dd = np.diff(p)         #Calculate difference between sequential points
        ddmod = np.mod(dd + discont, 2*discont) - discont   #Modulus 
        np.copyto(ddmod, discont, where=(ddmod == -discont) & (dd > 0))
        ph_correct = ddmod - dd
        np.copyto(ph_correct, 0, where=abs(dd) < discont)
        unWrapped = p
        unWrapped[1:] = p[1:] + ph_correct.cumsum()
        return unWrapped


class Probe():
    def __init__(self):
        self.xPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*0.5]) #[t,data] array!
        self.yPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*0.5]) #[t,data] array!
    
class Field():
    def __init__(self):
        self.B0 = 3 #Tesla
        self.xGradientWaveform = np.load(r"C:\Users\Oskar\Dropbox\Local files_oskars dator\Dropbox dokument\Python Scripts\General MRI\data\XGradientWaveform.npy")
        self.yGradientWaveform = np.load(r"C:\Users\Oskar\Dropbox\Local files_oskars dator\Dropbox dokument\Python Scripts\General MRI\data\XGradientWaveform.npy")
        self.xGradientWaveform[1,:]=np.roll(self.xGradientWaveform[1,:],-20)
        self.yGradientWaveform[1,:]=np.roll(self.yGradientWaveform[1,:],20)
        self.Gx_max = 0.1 #Tesla/m
        self.Gy_max = 0.1 #Tesla/m

        
    def setGrid(self,x0,x1,y0,y1):
        res = (x1-x0)/1000 #Resolution
        self.xPts = np.arange(x0,x1,res)
        self.yPts = np.arange(y0,y1,res)
        self.xGrid, self.yGrid = np.meshgrid(self.xPts,self.yPts, sparse=True)
        
    def setBField(self):
        
        Bgrid = self.B0 * np.ones(np.shape(self.xGrid))
        
#     def setBFieldTimeEvolution(self,xPos,yPos):
        
        
        

class Window(QtWidgets.QMainWindow, test1.Ui_Dialog):
    
    def __init__(self):
        
        self.Bfield = Field()
        self.FIDpointer = 0
        self.FIDs = []
        #Some setup:
        self.FIDs.append(MRSignal())

        super(self.__class__, self).__init__()
        self.setupUi(self)  # This is defined in design.py file automatically
        
        
        self.setWindowTitle("MR signals")
        self.checkBox_2.setChecked(True)
        self.pushButton.clicked.connect(self.updateParameters)
        self.pushButton_2.clicked.connect(self.togglePages)
        self.pushButton_3.clicked.connect(self.togglePages)
        self.pushButton_4.clicked.connect(self.addFID)

        self.region = pg.LinearRegionItem(values=(0,0.01))
        self.graphicsView_5.addItem(self.region, ignoreBounds=True)
        
        self.lineEdit_33.textChanged.connect(self.setRegionGraphics)
        self.lineEdit_34.textChanged.connect(self.setRegionGraphics)

        
        self.checkBox_5.stateChanged.connect(self.plot)
        self.checkBox_6.stateChanged.connect(self.plot)
        
        
        self.listWidget.currentRowChanged.connect(self.setCurrentFID)
        
        self.initLineEdit()
        self.setRegionGraphics()
        
        self.setupFIDs() #dummy
        
        
    def calculatePos(self):
        a=self.FIDs[0].diffPhase
        b=self.FIDs[1].diffPhase
        c=self.FIDs[2].diffPhase
        d=self.FIDs[3].diffPhase
        length = [len(a),len(b),len(c),len(d)]
        ind=np.min(length) #fixing some kind of rounding error
        diffSnippets = np.matrix([a[0:ind],b[0:ind],c[0:ind]]).transpose()
        r1 = [0.1,0.1,1]  #append 1
        r2 = [-0.1,0.1,1] #append 1
        r3 = [-0.1,-0.1,1]#append 1
        refPosMatrix = np.linalg.inv(np.matrix([r1,r2,r3]).transpose())
        FGg=np.multiply(1/self.FIDs[0].gyroMagneticRatio,np.matmul(diffSnippets,refPosMatrix))
        FG = FGg[:,0:2]
        Fg = FGg[:,2]
        FGplus = np.linalg.pinv(FG)
        solveInd = 3
        r = np.matmul(FGplus,(1/self.FIDs[0].gyroMagneticRatio) * np.matrix(self.FIDs[solveInd].diffPhase[0:ind]).transpose() - Fg)
        
        print(r)
    
    def setupFIDs(self):
        #Dummy function for setting up FIDs
        self.FIDs[0].t0 = 0.0017
        self.FIDs[0].t1 = 0.0019
        self.FIDpointer = len(self.FIDs)-1
        self.lineEdit_33.setText(str(self.FIDs[self.FIDpointer].t0 * 1e3))
        self.lineEdit_34.setText(str(self.FIDs[self.FIDpointer].t1 * 1e3))
        self.updateParameters()
        self.FIDs.append(MRSignal())
        self.listWidget.addItem('Snippet ' + str(len(self.FIDs)))
        self.FIDpointer = len(self.FIDs)-1
        self.FIDs[self.FIDpointer].Probe.xPos[1,:] = self.FIDs[self.FIDpointer].Probe.xPos[1,:]*(-1)
        self.FIDs[self.FIDpointer].Probe.yPos[1,:] = self.FIDs[self.FIDpointer].Probe.yPos[1,:]
        self.FIDs[self.FIDpointer].t0 = 0.0048
        self.FIDs[self.FIDpointer].t1 = 0.005
        self.lineEdit_33.setText(str(self.FIDs[self.FIDpointer].t0 * 1e3))
        self.lineEdit_34.setText(str(self.FIDs[self.FIDpointer].t1 * 1e3))
        self.updateParameters()
        self.FIDs.append(MRSignal())
        self.listWidget.addItem('Snippet ' + str(len(self.FIDs)))
        self.FIDpointer = len(self.FIDs)-1
        self.FIDs[self.FIDpointer].Probe.xPos[1,:] = self.FIDs[self.FIDpointer].Probe.xPos[1,:]*(-1)
        self.FIDs[self.FIDpointer].Probe.yPos[1,:] = self.FIDs[self.FIDpointer].Probe.yPos[1,:]*(-1)
        self.FIDs[self.FIDpointer].t0 = 0.0055
        self.FIDs[self.FIDpointer].t1 = 0.0057
        self.lineEdit_33.setText(str(self.FIDs[self.FIDpointer].t0 * 1e3))
        self.lineEdit_34.setText(str(self.FIDs[self.FIDpointer].t1 * 1e3))
        self.updateParameters()
        
        self.FIDs.append(MRSignal())
        self.listWidget.addItem('Snippet ' + str(len(self.FIDs)))
        self.FIDpointer = len(self.FIDs)-1
        self.FIDs[self.FIDpointer].Probe.xPos[1,:] = self.FIDs[self.FIDpointer].Probe.xPos[1,:]*0
        self.FIDs[self.FIDpointer].Probe.yPos[1,:] = self.FIDs[self.FIDpointer].Probe.yPos[1,:]*0
        self.FIDs[self.FIDpointer].t0 = 0.0055
        self.FIDs[self.FIDpointer].t1 = 0.0057
        self.lineEdit_33.setText(str(self.FIDs[self.FIDpointer].t0 * 1e3))
        self.lineEdit_34.setText(str(self.FIDs[self.FIDpointer].t1 * 1e3))
        self.updateParameters()
        
#     def setupProbes(self):
#         #Dummy function for setting up 3 static probe positions
#         for i in range(0,4):
#             self.FIDs[self.FIDpointer].Probe.append(Probe())
#             
#         self.FIDs[self.FIDpointer].Probe[1].xPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].xPos[1,:]*(-1)
#         self.FIDs[self.FIDpointer].Probe[1].yPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].yPos[1,:]
#         self.FIDs[self.FIDpointer].Probe[2].xPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].xPos[1,:]*(-1)
#         self.FIDs[self.FIDpointer].Probe[2].yPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].yPos[1,:]*(-1)
#         self.FIDs[self.FIDpointer].Probe[3].xPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].xPos[1,:]*0
#         self.FIDs[self.FIDpointer].Probe[3].yPos[1,:] = self.FIDs[self.FIDpointer].Probe[0].yPos[1,:]*0
    
    def setCurrentFID(self):
        self.FIDpointer=self.listWidget.currentRow()
        for i in np.arange(0,len(self.FIDs)):
            self.FIDs[i].RGB = 1000
        self.FIDs[self.FIDpointer].RGB = 5000
        self.setLineEdit()
    
    def addFID(self):
        self.FIDs.append(MRSignal())
        self.listWidget.addItem('Snippet ' + str(len(self.FIDs)))
        self.FIDpointer = len(self.FIDs)-1
        self.initLineEdit()
    
    def setRegionGraphics(self):
        self.region.setRegion([np.double(self.lineEdit_33.text()),np.double(self.lineEdit_34.text())])

        
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
        
        
        
        
    def zeroPad(self,array):
        zeros = np.zeros(self.FIDs[self.FIDpointer].Nbox)
        array = np.concatenate((array,zeros))
        array = np.concatenate((zeros,array))
        return array
        
    def LPfilter(self):
        if self.FIDs[self.FIDpointer].doLowPass:
            self.FIDs[self.FIDpointer].Nbox = np.int(self.lineEdit_15.text())
            self.FIDs[self.FIDpointer].I = self.zeroPad(self.FIDs[self.FIDpointer].I)
            self.FIDs[self.FIDpointer].Q = self.zeroPad(self.FIDs[self.FIDpointer].Q)
            conv = np.ones(self.FIDs[self.FIDpointer].Nbox)/self.FIDs[self.FIDpointer].Nbox
            self.FIDs[self.FIDpointer].I = np.convolve(conv,self.FIDs[self.FIDpointer].I,mode='valid')[np.int(self.FIDs[self.FIDpointer].Nbox/2):np.int(len(self.FIDs[self.FIDpointer].t)+self.FIDs[self.FIDpointer].Nbox/2)]
            self.FIDs[self.FIDpointer].Q = np.convolve(conv,self.FIDs[self.FIDpointer].Q,mode='valid')[np.int(self.FIDs[self.FIDpointer].Nbox/2):np.int(len(self.FIDs[self.FIDpointer].t)+self.FIDs[self.FIDpointer].Nbox/2)]
        

    def updateParameters(self):
        self.FIDs[self.FIDpointer].Nbox = np.int(self.lineEdit_15.text())
        self.FIDs[self.FIDpointer].addNoise = self.checkBox_9.isChecked()
        self.FIDs[self.FIDpointer].doLowPass = self.checkBox_2.isChecked()
        self.FIDs[self.FIDpointer].noiseLvl = np.double(self.lineEdit_31.text())
        self.FIDs[self.FIDpointer].T2 = np.double(self.lineEdit_32.text())
        self.FIDs[self.FIDpointer].f0 = np.double(self.lineEdit.text())
        self.FIDs[self.FIDpointer].dt = np.double(self.lineEdit_4.text())
        self.FIDs[self.FIDpointer].t0 = np.double(self.lineEdit_33.text())/1e3
        self.FIDs[self.FIDpointer].t1 = np.double(self.lineEdit_34.text())/1e3
        self.FIDs[self.FIDpointer].generateRFSignal(self.Bfield,self.FIDs[self.FIDpointer].Probe)
        self.LPfilter()
        self.FIDs[self.FIDpointer].calculatePhase()
        if len(self.FIDs)>3:
            self.calculatePos()
        self.plot()
        
    
    def plot(self):
        
    
        self.graphicsView.clear()
        self.graphicsView_2.clear()
        self.graphicsView_3.clear()
        self.graphicsView_4.clear()
        self.graphicsView_6.clear()
        self.graphicsView_7.clear()
        
        self.graphicsView.plot(self.FIDs[self.FIDpointer].t, self.FIDs[self.FIDpointer].RFsignal,pen=pg.mkPen('r', width=1))
        if self.checkBox_6.isChecked():
            self.graphicsView_2.plot(self.FIDs[self.FIDpointer].t, self.FIDs[self.FIDpointer].I,pen=pg.mkPen('r', width=1))
        if self.checkBox_5.isChecked():
            self.graphicsView_2.plot(self.FIDs[self.FIDpointer].t, self.FIDs[self.FIDpointer].Q,pen=pg.mkPen('b', width=1))
    
        self.graphicsView_3.plot(self.FIDs[self.FIDpointer].I,self.FIDs[self.FIDpointer].Q,pen=pg.mkPen('r', width=1))
        
        for x in self.FIDs:
            self.graphicsView_4.plot(x.t,x.phase,pen=pg.mkPen('r', width=1))
            self.graphicsView_4.plot(x.t,x.c2 * x.t**2 + x.c1*x.t + x.c0,pen=pg.mkPen('g', width=1))
            self.graphicsView_6.plot(x.fft_freq,20*np.log10(np.abs(x.fft[0:len(x.fft_freq)])),pen=pg.mkPen(x.RGB, width=1))
            self.graphicsView_7.plot(x.Probe.xPos[1,:],x.Probe.yPos[1,:], symbol='o',symbolBrush=x.RGB,pen=pg.mkPen(x.RGB, width=1))
#             self.graphicsView_7.plot(x.Probe.xPos[1,:],x.Probe.yPos[1,:])
            
        self.graphicsView_5.plot(self.Bfield.xGradientWaveform[0,:]*1e3,self.Bfield.xGradientWaveform[1,:],pen=pg.mkPen('r', width=1))
        self.graphicsView_5.plot(self.Bfield.yGradientWaveform[0,:]*1e3,self.Bfield.yGradientWaveform[1,:],pen=pg.mkPen('b', width=1))
        
    
    
    def setLineEdit(self):
        self.lineEdit.setText(str(self.FIDs[self.FIDpointer].f0)) #LO freq
        self.lineEdit_4.setText(str(self.FIDs[self.FIDpointer].dt))
        self.lineEdit_33.setText(str(self.FIDs[self.FIDpointer].t0 * 1e3))
        self.lineEdit_34.setText(str(self.FIDs[self.FIDpointer].t1 * 1e3))
        self.lineEdit_32.setText(str(self.FIDs[self.FIDpointer].T2))
        self.checkBox_9.setChecked(self.FIDs[self.FIDpointer].addNoise)
        self.checkBox_2.setChecked(self.FIDs[self.FIDpointer].doLowPass)
        self.lineEdit_31.setText(str(self.FIDs[self.FIDpointer].noiseLvl))
        self.plot()
        
        
    def initLineEdit(self):
        self.lineEdit.setText('126e6') #LO freq
        self.lineEdit_4.setText('0.5e-9')
#         self.lineEdit_11.setText('10e6')
        self.lineEdit_33.setText('0.0')
        self.lineEdit_34.setText('0.1')
        self.lineEdit_32.setText('500e-6')
        self.lineEdit_31.setText(str(self.FIDs[self.FIDpointer].noiseLvl))
        self.updateParameters()
        
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
    
