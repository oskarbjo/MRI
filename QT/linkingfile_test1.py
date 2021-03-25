
import test1

import sys
import matplotlib
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
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
        self.f0 = 126e6 #LO freq
        self.IF = 0
        self.larmor = 0
        self.larmor_end = 0
        self.dt = 0.5e-9
        self.dt_dec = 0.5e-7
        self.BW = 0
        self.Nsignals = 0
        self.T = 0
        self.t = [0]
        self.t_dec = [0]
        self.RFsignal = [0]
        self.I = [0]
        self.Q = [0]
        self.LO_I = [0]
        self.LO_Q = [0]
        self.phase = [0]
        self.diffPhase=[0]
        self.fft_freq=[0]
        self.fft=[0]
        self.c2, self.c1, self.c0 = [0,0,0]
        self.addNoise = True
        self.noiseLvl = 0.1
        self.T2 = 500e-6
        self.rollN = 0
        self.gyroMagneticRatio = 127.74e6/3 #Hz/T
        self.doLowPass = True
        self.RGB = 1000
        self.Nbox = 79
#         self.generateRFSignal()
        
    def generateRFSignal(self,Bfield,signalprobe,t0,t1):
        
        self.t = np.arange(0,(t1-t0),self.dt) #Always starts from zero for each new excitation!
        self.t_dec = np.arange(0,(t1-t0),self.dt_dec)
        self.t_dec = np.delete(self.t_dec, len(self.t_dec) - 1) #Removes last element to ensure that interpolation is not out of range
        Gx_temp=interp1d(Bfield.xGradientWaveform[0,:]*1e-3,Bfield.xGradientWaveform[1,:])
        Gy_temp=interp1d(Bfield.yGradientWaveform[0,:]*1e-3,Bfield.yGradientWaveform[1,:])
        xPos_temp=interp1d(signalprobe.xPos[0,:],signalprobe.xPos[1,:])
        yPos_temp=interp1d(signalprobe.yPos[0,:],signalprobe.yPos[1,:])
        Gx_interp = Gx_temp(self.t+t0)
        Gy_interp = Gy_temp(self.t+t0)
        xPos_interp = xPos_temp(self.t+t0)
        yPos_interp = yPos_temp(self.t+t0)
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
        self.LPfilter()
        I_temp = interp1d(self.t,self.I)
        Q_temp = interp1d(self.t, self.Q)
        self.I_dec = I_temp(self.t_dec)
        self.Q_dec = Q_temp(self.t_dec)
        self.calculateFFTs()
        self.calculatePhase()

    def LPfilter(self):
        if self.doLowPass:
            # self.snippets[self.snippetPointer].probes[self.probePointer].FID.I = self.zeroPad(self.snippets[self.snippetPointer].probes[self.probePointer].FID.I)
            # self.snippets[self.snippetPointer].probes[self.probePointer].FID.Q = self.zeroPad(self.snippets[self.snippetPointer].probes[self.probePointer].FID.Q)
            conv = np.ones(self.Nbox) / self.Nbox
            I_temp = np.convolve(conv, self.I)
            Q_temp = np.convolve(conv, self.Q)
            self.I = I_temp[np.int(np.round(self.Nbox / 2)):np.int(np.round(self.Nbox / 2)) + len(self.I)]
            self.Q = Q_temp[np.int(np.round(self.Nbox / 2)):np.int(np.round(self.Nbox / 2)) + len(self.Q)]

    def calculatePhase(self):
#         self.phase = np.unwrap(2*np.arctan(self.Q/self.I))/2
        self.phase = self.unWrapPhaseSimple(np.arctan(self.Q_dec/self.I_dec))
        self.differentiatePhase()
        
    def differentiatePhase(self):
        self.diffPhase=np.diff(self.phase)/self.dt_dec
        A = np.vstack([self.t_dec**2, self.t_dec, np.ones(len(self.t_dec))]).T
        self.c2, self.c1, self.c0 = np.linalg.lstsq(A, self.phase, rcond=None)[0]
        self.diffPhase_fit = np.diff(np.power(self.t_dec,2)*self.c2 + self.t_dec*self.c1 + self.c0)/self.dt_dec

        N=15
        self.diffPhase_filtered = np.zeros(len(self.diffPhase)-N)
        self.diffPhase_fit_filtered = np.zeros(len(self.diffPhase_fit) - N)
        for i in np.arange(0,len(self.diffPhase_filtered)): #Multiplying with matrix F
            self.diffPhase_filtered[i] = (self.diffPhase[i] - self.diffPhase[i+5] + self.diffPhase[i+10] - self.diffPhase[i+15])/4
            self.diffPhase_fit_filtered[i] = (self.diffPhase_fit[i] - self.diffPhase_fit[i + 5] + self.diffPhase_fit[i + 10] -
                                            self.diffPhase_fit[i + 15]) / 4
        print(' ')

    def calculateFFTs(self):
        self.fft_freq = np.linspace(0,0.5/(self.dt_dec),np.int(len(self.I_dec)/2))
        self.fft = np.fft.fft(self.I_dec)
    
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
        self.FID = MRSignal()
        
    def setPositions(self,x,y):
        self.xPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*x]) #[t,data] array!
        self.yPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*y]) #[t,data] array!
        
class Field():
    def __init__(self):
        self.B0 = 3 #Tesla
        self.xGradientWaveform = np.load(r"C:\Users\Oskar\Dropbox\Local files_oskars dator\Dropbox dokument\Python Scripts\General MRI\data\XGradientWaveform.npy")
        self.yGradientWaveform = np.load(r"C:\Users\Oskar\Dropbox\Local files_oskars dator\Dropbox dokument\Python Scripts\General MRI\data\XGradientWaveform.npy")
        self.xGradientWaveform[0,:]=self.xGradientWaveform[0,:]*1000
        self.yGradientWaveform[0,:]=self.yGradientWaveform[0,:]*1000
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
        
class Snippet():
    
    def __init__(self,t0,t1):
        self.t0 = t0
        self.t1 = t1
        self.probes = []
        
        

class Window(QtWidgets.QMainWindow, test1.Ui_Dialog):
    
    def __init__(self):

        super(self.__class__, self).__init__()
        self.setupUi(self)  # This is defined in design.py file automatically
        self.initLineEdit()
        self.Bfield = Field()
        self.snippetPointer = 0
        self.probePointer = 0
        self.snippets = []
        self.initSnippet()

        
        self.setWindowTitle("MR signals")
        self.checkBox_2.setChecked(True)
        self.pushButton.clicked.connect(self.updateParameters)
        self.pushButton_2.clicked.connect(self.togglePages)
        self.pushButton_3.clicked.connect(self.togglePages)
        self.pushButton_4.clicked.connect(self.addSnippet)
        self.pushButton_6.clicked.connect(self.addProbe)
        self.pushButton_8.clicked.connect(self.calculatePos)
        self.region = pg.LinearRegionItem(values=(0,0.01))
        self.graphicsView_5.addItem(self.region, ignoreBounds=True)
        
        self.lineEdit_33.textChanged.connect(self.setRegionGraphics)
        self.lineEdit_34.textChanged.connect(self.setRegionGraphics)
        
        self.lineEdit_35.textChanged.connect(self.setxPosition)
        self.lineEdit_36.textChanged.connect(self.setyPosition)

        
        self.checkBox_5.stateChanged.connect(self.plot)
        self.checkBox_6.stateChanged.connect(self.plot)
        
        
        self.listWidget.currentRowChanged.connect(self.setCurrentSnippet)
        self.listWidget_2.currentRowChanged.connect(self.setCurrentProbe)
        self.setTabOrder(self.lineEdit_33, self.lineEdit_34)
        
        
        self.initLineEdit()
        self.setRegionGraphics()
        self.setupSnippets()
        self.setupProbes()
        self.updateAll()


    def updateAll(self):
        for i in np.arange(0,len(self.snippets)):
            ind=0
            for j in np.arange(0,len(self.snippets[-1].probes)):
                self.probePointer = j
                self.snippetPointer = i
                self.setLineEdit()
                self.updateParameters()
        
    def setxPosition(self):
        x = np.double(self.lineEdit_35.text())
        y = self.snippets[self.snippetPointer].probes[self.probePointer].yPos[1,0]
        for i in self.snippets:
            i.probes[self.probePointer].xPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*x])
            i.probes[self.probePointer].yPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*y])
        
    def setyPosition(self):
        y = np.double(self.lineEdit_36.text())
        x = self.snippets[self.snippetPointer].probes[self.probePointer].xPos[1,0]
        for i in self.snippets:
            i.probes[self.probePointer].xPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*x])
            i.probes[self.probePointer].yPos = np.asarray([np.linspace(0,25e-3,100),np.ones(100)*y])
        
    def calculatePos(self):
        length = len(self.snippets[0].probes[0].FID.diffPhase_fit_filtered) + len(
            self.snippets[1].probes[0].FID.diffPhase_fit_filtered) + len(
            self.snippets[2].probes[0].FID.diffPhase_fit_filtered)
        mat = np.matrix((np.zeros(length), np.zeros(length), np.zeros(length)))

        for i in np.arange(0, len(self.snippets[0].probes) - 1):
            a = self.snippets[0].probes[i].FID.diffPhase_fit_filtered
            b = self.snippets[1].probes[i].FID.diffPhase_fit_filtered
            c = self.snippets[2].probes[i].FID.diffPhase_fit_filtered
            conc = np.concatenate((a, b, c))
            mat[i, :] = conc
        diffSnippets = mat.transpose()

        x1 = self.snippets[0].probes[0].xPos[1, 0]
        x2 = self.snippets[0].probes[1].xPos[1, 0]
        x3 = self.snippets[0].probes[2].xPos[1, 0]
        y1 = self.snippets[0].probes[0].yPos[1, 0]
        y2 = self.snippets[0].probes[1].yPos[1, 0]
        y3 = self.snippets[0].probes[2].yPos[1, 0]
        r1 = [x1, y1, 1]  # append 1
        r2 = [x2, y2, 1]  # append 1
        r3 = [x3, y3, 1]  # append 1
        refPosMatrix = np.linalg.inv(np.matrix([r1, r2, r3]).transpose())
        FGg = np.multiply(1 / self.snippets[0].probes[0].FID.gyroMagneticRatio, np.matmul(diffSnippets, refPosMatrix))
        FG = FGg[:, 0:2]
        Fg = FGg[:, 2]
        FGplus = np.linalg.pinv(FG)
        solveInd = 3
        a = self.snippets[0].probes[solveInd].FID.diffPhase_fit_filtered
        b = self.snippets[1].probes[solveInd].FID.diffPhase_fit_filtered
        c = self.snippets[2].probes[solveInd].FID.diffPhase_fit_filtered
        conc2 = np.concatenate((a, b, c))
        conc2 = np.matrix(conc2.transpose())
        Dfi = conc2.transpose()
        plt.figure()
        plt.plot(Dfi)
        plt.show()
        r = np.matmul(FGplus, (1 / self.snippets[0].probes[0].FID.gyroMagneticRatio) * np.matrix(Dfi - Fg))

        print(r)
    
    def setupSnippets(self): #dummy function
        self.addSnippet()
        self.addSnippet()
        self.snippets[0].t0=0.001
        self.snippets[0].t1=0.0015
        self.snippets[1].t0=0.005
        self.snippets[1].t1=0.0055
        self.snippets[2].t0=0.009
        self.snippets[2].t1=0.0095


        
    def setupProbes(self):  #dummy function
        self.addProbe()
        self.addProbe()
        self.addProbe()
        datax=[0.1,-0.1,-0.1,0.1]
        datay=[0.1,0.1,-0.1,-0.1]
        for i in self.snippets:
            ind=0
            for j in i.probes:
                j.setPositions(datax[ind],datay[ind])
                ind=ind+1
        
    def addProbe(self):
        a=self.snippetPointer
        for i in np.arange(0,len(self.snippets)):
            self.snippets[i].probes.append(Probe())
            self.snippetPointer=i
            self.probePointer=len(self.snippets[i].probes)-1
            self.setLineEdit()
            self.updateParameters()
        self.snippetPointer=a
        self.probePointer = len(self.snippets[0].probes)-1
        self.listWidget_2.addItem('Probe ' + str(len(self.snippets[0].probes)))
        self.plot()
        
    def setCurrentProbe(self):
        self.probePointer = self.listWidget_2.currentRow()
        self.setLineEdit()
        self.updateRegionGraphics()
        self.plot()
        
    def setCurrentSnippet(self):
        self.snippetPointer=self.listWidget.currentRow()
        self.setLineEdit()
        self.updateRegionGraphics()
        self.plot()
    
    def initSnippet(self):
        self.snippets.append(Snippet(0,0.0001))
        self.listWidget.addItem('Snippet ' + str(len(self.snippets)))
        self.snippetPointer = len(self.snippets)-1
        self.snippets[self.snippetPointer].probes.append(Probe())
        self.probePointer = len(self.snippets[self.snippetPointer].probes)-1
        self.setLineEdit()
        self.updateParameters()
        self.plot()
        
    def addSnippet(self):
        self.snippets.append(Snippet(0,0.0001))
        self.listWidget.addItem('Snippet ' + str(len(self.snippets)))
        self.snippetPointer = len(self.snippets)-1
#         self.listWidget.setCurrentRow(self.snippetPointer)
        a=self.probePointer
        for i in self.snippets[0].probes:
            self.snippets[self.snippetPointer].probes.append(Probe())
            self.probePointer = len(self.snippets[self.snippetPointer].probes)-1
            self.setLineEdit()
            self.updateParameters()
        self.probePointer=a
#         self.snippets[self.snippetPointer].probes[self.probePointer].FID.generateRFSignal(self.Bfield, self.snippets[self.snippetPointer].probes[self.probePointer])
        self.setRegionGraphics()
        self.plot()

    def updateRegionGraphics(self):
        self.region.setRegion([self.snippets[self.snippetPointer].t0*1000, self.snippets[self.snippetPointer].t1*1000])

    def setRegionGraphics(self):
        t0 = np.double(self.lineEdit_33.text())
        t1 = np.double(self.lineEdit_34.text())
        self.region.setRegion([t0,t1])
        self.snippets[self.snippetPointer].t0 = t0/1000
        self.snippets[self.snippetPointer].t1 = t1/1000
        
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
        zeros = np.zeros(self.snippets[self.snippetPointer].probes[self.probePointer].FID.Nbox)
        array = np.concatenate((array,zeros))
        array = np.concatenate((zeros,array))
        return array
        


    def updateParameters(self):
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.Nbox = np.int(self.lineEdit_15.text())
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.addNoise = self.checkBox_9.isChecked()
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.doLowPass = self.checkBox_2.isChecked()
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.noiseLvl = np.double(self.lineEdit_31.text())
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.T2 = np.double(self.lineEdit_32.text())
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.f0 = np.double(self.lineEdit.text())
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.dt = np.double(self.lineEdit_4.text())
        self.snippets[self.snippetPointer].t0 = np.double(self.lineEdit_33.text())/1000
        self.snippets[self.snippetPointer].t1 = np.double(self.lineEdit_34.text())/1000
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.Nbox = np.int(self.lineEdit_15.text())
        t0 = self.snippets[self.snippetPointer].t0
        t1 = self.snippets[self.snippetPointer].t1
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.generateRFSignal(self.Bfield,self.snippets[self.snippetPointer].probes[self.probePointer],t0,t1)
        self.snippets[self.snippetPointer].probes[self.probePointer].FID.calculatePhase()
        
        
        self.plot()
        
    
    def plot(self):
        
    
        self.graphicsView.clear()
        self.graphicsView_2.clear()
        self.graphicsView_3.clear()
        self.graphicsView_4.clear()
        self.graphicsView_6.clear()
        self.graphicsView_7.clear()
        
        self.graphicsView.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.t, self.snippets[self.snippetPointer].probes[self.probePointer].FID.RFsignal,pen=pg.mkPen('r', width=1))
        if self.checkBox_6.isChecked():
            self.graphicsView_2.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.t, self.snippets[self.snippetPointer].probes[self.probePointer].FID.I,pen=pg.mkPen('r', width=1))
        if self.checkBox_5.isChecked():
            self.graphicsView_2.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.t, self.snippets[self.snippetPointer].probes[self.probePointer].FID.Q,pen=pg.mkPen('b', width=1))
    
        self.graphicsView_3.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.I,self.snippets[self.snippetPointer].probes[self.probePointer].FID.Q,pen=pg.mkPen('r', width=1))
        
        
        self.graphicsView_4.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.t_dec,self.snippets[self.snippetPointer].probes[self.probePointer].FID.phase,pen=pg.mkPen(self.snippets[self.snippetPointer].probes[self.probePointer].FID.RGB, width=1))
#         self.graphicsView_4.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.t,self.snippets[self.snippetPointer].probes[self.probePointer].FID.c2 * self.snippets[self.snippetPointer].probes[self.probePointer].FID.t**2 + self.snippets[self.snippetPointer].probes[self.probePointer].FID.c1*self.snippets[self.snippetPointer].probes[self.probePointer].FID.t + self.snippets[self.snippetPointer].probes[self.probePointer].FID.c0,pen=pg.mkPen(self.snippets[self.snippetPointer].probes[self.probePointer].FID.RGB, width=1))
        self.graphicsView_6.plot(self.snippets[self.snippetPointer].probes[self.probePointer].FID.fft_freq,20*np.log10(np.abs(self.snippets[self.snippetPointer].probes[self.probePointer].FID.fft[0:len(self.snippets[self.snippetPointer].probes[self.probePointer].FID.fft_freq)])),pen=pg.mkPen(self.snippets[self.snippetPointer].probes[self.probePointer].FID.RGB, width=1))
        self.graphicsView_7.plot(self.snippets[self.snippetPointer].probes[self.probePointer].xPos[1,:],self.snippets[self.snippetPointer].probes[self.probePointer].yPos[1,:], symbol='o',symbolBrush=self.snippets[self.snippetPointer].probes[self.probePointer].FID.RGB,pen=pg.mkPen(self.snippets[self.snippetPointer].probes[self.probePointer].FID.RGB, width=1))
#             self.graphicsView_7.plot(x.Probe.xPos[1,:],x.Probe.yPos[1,:])
            
        self.graphicsView_5.plot(self.Bfield.xGradientWaveform[0,:],self.Bfield.xGradientWaveform[1,:],pen=pg.mkPen('r', width=1))
        self.graphicsView_5.plot(self.Bfield.yGradientWaveform[0,:],self.Bfield.yGradientWaveform[1,:],pen=pg.mkPen('b', width=1))
        
    
    
    def setLineEdit(self):
        self.lineEdit.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].FID.f0)) #LO freq
        self.lineEdit_4.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].FID.dt))
        self.lineEdit_33.blockSignals(True)
        self.lineEdit_34.blockSignals(True)
        self.lineEdit_33.setText(str(self.snippets[self.snippetPointer].t0*1000))
        self.lineEdit_34.setText(str(self.snippets[self.snippetPointer].t1*1000))
        self.lineEdit_33.blockSignals(False)
        self.lineEdit_34.blockSignals(False)
        self.lineEdit_32.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].FID.T2))
        self.checkBox_9.setChecked(self.snippets[self.snippetPointer].probes[self.probePointer].FID.addNoise)
        self.checkBox_2.setChecked(self.snippets[self.snippetPointer].probes[self.probePointer].FID.doLowPass)
        self.lineEdit_31.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].FID.noiseLvl))
        self.lineEdit_35.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].xPos[1,0]))
        self.lineEdit_36.setText(str(self.snippets[self.snippetPointer].probes[self.probePointer].yPos[1,0]))
    
        
    def initLineEdit(self):
        self.lineEdit.setText('126e6') #LO freq
        self.lineEdit_4.setText('0.5e-9')
#         self.lineEdit_11.setText('10e6')
#         self.lineEdit_33.setText('0.0')
#         self.lineEdit_34.setText('0.1')
        self.lineEdit_32.setText('500e-6')
        self.lineEdit_31.setText('0.1')

        
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
    
