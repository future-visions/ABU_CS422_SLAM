from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ekf import predict
import numpy as np
from numpy import sin, cos, pi
import matplotlib
matplotlib.use('QT5Agg')

class PlotW(QWidget):
    def __init__(self):
        super(PlotW,self).__init__()
        self.setLayout()
        self.t = np.linspace(0, 1, 1000)
        self.MEAN=0
        self.VARIANCE=0
        self.LENGTH=0
        self.state=0

    def setLayout(self):
        self.fig = Figure((18,5), 100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [])
        self.line2, = self.ax.plot([], [],  linestyle='None', marker="x", markersize=5)
        self.line3, = self.ax.plot([], [])
        self.fig.legend(["Actual", "Measurements", "Prediction"])
        self.fig.tight_layout()
        self.pw=FigureCanvas(self.fig)
        self.graphbox = QVBoxLayout(self)
        self.graphbox.addWidget(self.pw)

    def setData1(self, x, y):
        self.line.set_data(x,y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setData2(self, x, y):
        self.line2.set_data(x,y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setData3(self, x, y):
        self.line3.set_data(x,y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setP(self,m,v,l):
        self.MEAN=np.float(m)
        self.VARIANCE=np.float(v)
        self.LENGTH=np.int(l)
        self.t = np.linspace(0, 1, self.LENGTH)


    def evaluate(self, s):
        t=self.t
        self.state=0
        f=eval(s)
        self.func=f
        self.set_measurements()

    def set_measurements(self):
        self.measurements=self.func+np.random.normal(self.MEAN, self.VARIANCE, self.LENGTH)
        x = np.zeros((2, 1))  # initial state (location and velocity)
        P = np.array([[1., 0.],
                      [0., 0.]]) * 10  # initial variance
        u = np.zeros((2, self.LENGTH))
        u[0] = self.func  # external motion
        u[1] = np.full((1, self.LENGTH), 1 / self.LENGTH)
        F = np.array([[1., 0.],
                      [0., 0.]])  # next state function
        H = np.array([[1., 0.]])  # measurement function
        R = np.array([[1.]]) * self.VARIANCE  # measurement variance

        x_n = x
        p_n = P
        x_predict = []
        for i, m in enumerate(self.measurements):
            x_n, p_n = predict(x, u[:, [i]], m, F, p_n, R, 10000, H)
            x_predict.append(x_n[0, 0])
        self.xP=np.array(x_predict)

    def nextState(self):
        self.state+=1
        self.setData1(self.t[:self.state], self.func[:self.state])
        self.setData2(self.t[:self.state], self.measurements[:self.state])
        self.setData3(self.t[:self.state], self.xP[:self.state])


class BaseWidget(QWidget):
    def __init__(self):
        super(BaseWidget,self).__init__()
        self.setLayout()
    def setLayout(self):
        self.lay = QVBoxLayout(self)
    def addwidget(self, w):
        self.lay.addWidget(w)
        self.show()

class ControlWidget(QWidget):
    def __init__(self, parent):
        super(ControlWidget,self).__init__()
        self.state = 0
        self.length_=0
        self.timer = QTimer(self)
        self.setLayout()
        self.parent=parent
    def setLayout(self):
        self.lay = QHBoxLayout(self)
        self.lbl1 = QLabel("Function")
        self.te1 = QLineEdit(self)
        self.te2 = QLineEdit(self)
        self.te2.setMaximumWidth(50)
        self.btn1 = QPushButton("Eval")
        self.lbl2 = QLabel("Mean")
        self.lbl3 = QLabel("Variance")
        self.lbl5 = QLabel("Length")
        self.lbl6 = QLabel("State: %d  "%(self.state))
        self.te3 = QLineEdit(self)
        self.te4 = QLineEdit(self)
        self.te5 = QLineEdit(self)
        self.lbl4 = QLabel("          Speed")
        self.btn2 = QPushButton("Next State")
        self.btn3 = QPushButton("Automate")
        self.te3.setMaximumWidth(30)
        self.te4.setMaximumWidth(30)
        self.te5.setMaximumWidth(30)
        self.lay.addWidget(self.lbl2)
        self.lay.addWidget(self.te3)
        self.lay.addWidget(self.lbl3)
        self.lay.addWidget(self.te4)
        self.lay.addWidget(self.lbl5)
        self.lay.addWidget(self.te5)
        self.lay.addWidget(self.lbl1)
        self.lay.addWidget(self.te1)
        self.lay.addWidget(self.btn1)
        self.lay.addWidget(self.lbl6)
        self.lay.addWidget(self.btn2)
        self.lay.addWidget(self.lbl4)
        self.lay.addWidget(self.te2)
        self.lay.addWidget(self.btn3)
        self.btn1.clicked.connect(self.click_eval)
        self.btn2.clicked.connect(self.click_next)
        self.btn3.clicked.connect(self.click_auto)

    def click_eval(self):
        s=self.te1.text()
        mean = self.te3.text()
        variance = self.te4.text()
        self.length_ = self.te5.text()
        self.parent.setP(mean,variance,self.length_)
        self.parent.evaluate(s)

    def click_next(self):
        self.parent.nextState()
        self.state += 1
        self.update_state()
        if self.state == int(self.length_):
            self.btn2.setEnabled(False)
            self.timer.stop()

    def update_state(self):
        self.lbl6.setText("State: %d  "%(self.state))

    def click_auto(self):
        interv = self.te2.text()
        self.timer.setInterval(int(interv))
        self.timer.timeout.connect(self.click_next)
        self.timer.start()


class ControlWidget2(QWidget):
    def __init__(self, friend):
        super(ControlWidget2,self).__init__()
        self.setLayout()
        self.friend=friend
    def setLayout(self):
        self.lay = QHBoxLayout(self)
        self.btn1 = QPushButton("Clear")
        self.btn2 = QPushButton("Show All")
        self.btn3 = QPushButton("Stop")
        self.lay.addWidget(self.btn1)
        self.lay.addWidget(self.btn2)
        self.lay.addWidget(self.btn3)
        self.btn2.clicked.connect(self.click_showAll)
        self.btn1.clicked.connect(self.click_clear)
        self.btn3.clicked.connect(self.click_stop)

    def click_showAll(self):
        xp=self.friend.parent.xP
        m=self.friend.parent.measurements
        f=self.friend.parent.func
        t=self.friend.parent.t
        self.friend.parent.setData1(t,f)
        self.friend.parent.setData2(t, m)
        self.friend.parent.setData3(t, xp)

    def click_clear(self):
        self.friend.te1.setText("")
        self.friend.te2.setText("")
        self.friend.te3.setText("")
        self.friend.te4.setText("")
        self.friend.te5.setText("")
        self.friend.state=0
        self.friend.update_state()
        self.friend.btn2.setEnabled(True)
        self.friend.parent.setData1([],[])
        self.friend.parent.setData2([],[])
        self.friend.parent.setData3([],[])

    def click_stop(self):
        self.friend.timer.stop()

app = QApplication([])
b=BaseWidget()
p=PlotW()
c=ControlWidget(p)
c2=ControlWidget2(c)
b.addwidget(p)
b.addwidget(c)
b.addwidget(c2)
app.exec_()