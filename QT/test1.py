# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test1.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1587, 983)
        self.stackedWidget = QtWidgets.QStackedWidget(Dialog)
        self.stackedWidget.setGeometry(QtCore.QRect(10, 10, 1591, 971))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setEnabled(True)
        self.page.setObjectName("page")
        self.graphicsView_2 = PlotWidget(self.page)
        self.graphicsView_2.setGeometry(QtCore.QRect(0, 180, 800, 150))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.groupBox = QtWidgets.QGroupBox(self.page)
        self.groupBox.setGeometry(QtCore.QRect(20, 670, 431, 151))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(130, 70, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_4.setGeometry(QtCore.QRect(170, 70, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(60, 30, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_2.setGeometry(QtCore.QRect(250, 50, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")
        self.lineEdit_15 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_15.setGeometry(QtCore.QRect(60, 70, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_15.setFont(font)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 70, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.checkBox_9 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_9.setGeometry(QtCore.QRect(250, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.checkBox_9.setFont(font)
        self.checkBox_9.setObjectName("checkBox_9")
        self.lineEdit_31 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_31.setGeometry(QtCore.QRect(60, 110, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_31.setFont(font)
        self.lineEdit_31.setObjectName("lineEdit_31")
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        self.label_23.setGeometry(QtCore.QRect(10, 110, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.lineEdit_32 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_32.setGeometry(QtCore.QRect(170, 30, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_32.setFont(font)
        self.lineEdit_32.setObjectName("lineEdit_32")
        self.label_24 = QtWidgets.QLabel(self.groupBox)
        self.label_24.setGeometry(QtCore.QRect(130, 30, 21, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.graphicsView = PlotWidget(self.page)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(0, 20, 800, 150))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_5 = PlotWidget(self.page)
        self.graphicsView_5.setGeometry(QtCore.QRect(0, 340, 800, 150))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.lineEdit_34 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_34.setGeometry(QtCore.QRect(720, 390, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_34.setFont(font)
        self.lineEdit_34.setObjectName("lineEdit_34")
        self.lineEdit_33 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_33.setGeometry(QtCore.QRect(720, 350, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_33.setFont(font)
        self.lineEdit_33.setObjectName("lineEdit_33")
        self.listWidget = QtWidgets.QListWidget(self.page)
        self.listWidget.setGeometry(QtCore.QRect(460, 690, 171, 101))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.listWidget.setFont(font)
        self.listWidget.setObjectName("listWidget")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setGeometry(QtCore.QRect(460, 790, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.page)
        self.pushButton_5.setGeometry(QtCore.QRect(560, 790, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton = QtWidgets.QPushButton(self.page)
        self.pushButton.setGeometry(QtCore.QRect(20, 830, 56, 17))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.checkBox_6 = QtWidgets.QCheckBox(self.page)
        self.checkBox_6.setEnabled(True)
        self.checkBox_6.setGeometry(QtCore.QRect(640, 190, 53, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.checkBox_6.setFont(font)
        self.checkBox_6.setChecked(True)
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_5 = QtWidgets.QCheckBox(self.page)
        self.checkBox_5.setGeometry(QtCore.QRect(710, 190, 53, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.checkBox_5.setFont(font)
        self.checkBox_5.setObjectName("checkBox_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.page)
        self.groupBox_2.setGeometry(QtCore.QRect(820, -10, 751, 971))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.graphicsView_6 = PlotWidget(self.groupBox_2)
        self.graphicsView_6.setGeometry(QtCore.QRect(10, 30, 700, 150))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.graphicsView_4 = PlotWidget(self.groupBox_2)
        self.graphicsView_4.setGeometry(QtCore.QRect(365, 190, 345, 345))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_3 = PlotWidget(self.groupBox_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(10, 190, 345, 345))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_7 = PlotWidget(self.groupBox_2)
        self.graphicsView_7.setGeometry(QtCore.QRect(10, 540, 345, 345))
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.lineEdit_35 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_35.setGeometry(QtCore.QRect(280, 790, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_35.setFont(font)
        self.lineEdit_35.setText("")
        self.lineEdit_35.setObjectName("lineEdit_35")
        self.lineEdit_36 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_36.setGeometry(QtCore.QRect(280, 830, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.lineEdit_36.setFont(font)
        self.lineEdit_36.setText("")
        self.lineEdit_36.setObjectName("lineEdit_36")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(360, 860, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.listWidget_2 = QtWidgets.QListWidget(self.groupBox_2)
        self.listWidget_2.setGeometry(QtCore.QRect(360, 760, 171, 101))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.listWidget_2.setFont(font)
        self.listWidget_2.setObjectName("listWidget_2")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_2.addItem(item)
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(460, 860, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_8.setGeometry(QtCore.QRect(170, 910, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.label_25 = QtWidgets.QLabel(self.page_2)
        self.label_25.setGeometry(QtCore.QRect(800, 40, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.page_2)
        self.label_26.setGeometry(QtCore.QRect(800, 80, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.page_2)
        self.label_27.setGeometry(QtCore.QRect(920, 40, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.page_2)
        self.label_28.setGeometry(QtCore.QRect(920, 80, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.stackedWidget.addWidget(self.page_2)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setGeometry(QtCore.QRect(560, 0, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(650, 0, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Snippet parameters"))
        self.label_5.setText(_translate("Dialog", "RF dt"))
        self.lineEdit_4.setText(_translate("Dialog", "0.5e-9"))
        self.label_2.setText(_translate("Dialog", "LO Freq"))
        self.lineEdit.setText(_translate("Dialog", "20e6"))
        self.checkBox_2.setText(_translate("Dialog", "Lowpass filter"))
        self.lineEdit_15.setText(_translate("Dialog", "79"))
        self.label_11.setText(_translate("Dialog", "Boxcar N"))
        self.checkBox_9.setText(_translate("Dialog", "Add noise"))
        self.lineEdit_31.setText(_translate("Dialog", "5"))
        self.label_23.setText(_translate("Dialog", "Noise lvl"))
        self.lineEdit_32.setText(_translate("Dialog", "50e-6"))
        self.label_24.setText(_translate("Dialog", "T2"))
        self.lineEdit_34.setText(_translate("Dialog", "0.1"))
        self.lineEdit_33.setText(_translate("Dialog", "0"))
        self.pushButton_4.setText(_translate("Dialog", "Add snippet"))
        self.pushButton_5.setText(_translate("Dialog", "Remove snippet"))
        self.pushButton.setText(_translate("Dialog", "Update"))
        self.checkBox_6.setText(_translate("Dialog", "I"))
        self.checkBox_5.setText(_translate("Dialog", "Q"))
        self.groupBox_2.setTitle(_translate("Dialog", "Overview"))
        self.pushButton_6.setText(_translate("Dialog", "Add probe"))
        __sortingEnabled = self.listWidget_2.isSortingEnabled()
        self.listWidget_2.setSortingEnabled(False)
        item = self.listWidget_2.item(0)
        item.setText(_translate("Dialog", "Probe 1"))
        self.listWidget_2.setSortingEnabled(__sortingEnabled)
        self.pushButton_7.setText(_translate("Dialog", "Remove probe"))
        self.pushButton_8.setText(_translate("Dialog", "Calculate"))
        self.label_25.setText(_translate("Dialog", "Signal 1 t0 = "))
        self.label_26.setText(_translate("Dialog", "Signal 1 t1 = "))
        self.label_27.setText(_translate("Dialog", "Signal 2 t0 = "))
        self.label_28.setText(_translate("Dialog", "Signal 2 t1 = "))
        self.pushButton_2.setText(_translate("Dialog", "Page 1"))
        self.pushButton_3.setText(_translate("Dialog", "Page 2"))
from pyqtgraph import PlotWidget
