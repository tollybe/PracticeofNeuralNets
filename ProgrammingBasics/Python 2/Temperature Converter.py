"""
import sys

from PyQt4 import QtCore, QtGui, uic


form_class = uic.loadUiType("MyFirstGUI.ui")[0]

class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self,parent = None):
        QtGui.QMainWindow.__init__(self,parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.button_clicked)

    def button_clicked(self):
        x = self.pushButton.x()
        y = self.pushButton.y()
        x += 50
        y += 50
        self.pushButton.move(x,y)


app = QtGui.QApplication(sys.argv)
myWindow = MyWindowClass()
myWindow.show()
app.exec_()

"""
import sys
from PyQt4 import QtCore, QtGui, uic

form_class = uic.loadUiType("tempconv.ui") [0]


class MyWindowClass(QtGui.QMainWindow,form_class):
    def __init__(self, parent = None):
        QtGui.QMainWindow.__init__(self,parent)
        self.setupUi(self)
        self.btnCtoF.clicked.connect(self.btn_CtoF_clicked)
        self.btnFtoC.clicked.connect(self.btn_FtoC_clicked)
         #menu items
        self.actionC_to_F.triggered.connect(self.btn_CtoF_clicked)
        self.actionF_to_C.triggered.connect(self.btn_FtoC_clicked)
        self.actionExit.triggered.connect(self.menuExit_selected)

    def btn_CtoF_clicked(self):
        cel = float(self.editCel.text())
        fahr = cel * 9.0 / 5 + 32
        self.spinFahr.setValue(int(fahr + 0.5))

    def btn_FtoC_clicked(self):
        fahr = self.spinFahr.value()
        cel = (fahr - 32) * 5/9.0
        self.editCel.setText(str(cel))

    def menuExit_selected(self):
        self.close()


app = QtGui.QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
app.exec_()
