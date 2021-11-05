from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# from PyQt5.QtCore import Qt, QUrl
# from PyQt5.QtGui import QPalette
from PyQt5.uic import loadUi
from GUI_media import CMultiMedia # Qt에서 제공하는 멀티미디어 라이브러리 사용. PyQt5나 PySide2 등을 통해 사용 가능
import sys
import datetime

from models import predict


class CWidget(QWidget):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self) # Qt Designer로 생성한 ui파일 load
 
        # Multimedia Object, view = QVideoWidget
        self.mp = CMultiMedia(self, self.view) # media.py에 선언된 CMultiMedia 객체 생성...
        self.video_path = None

        # video background color
        pal = QPalette() # 각 위젯 상태에 대한 색상 그룹을 포함한다.
        pal.setColor(QPalette.Background, Qt.black) # (cr, color)
        self.view.setAutoFillBackground(True) # 
        self.view.setPalette(pal) # 검은색으로 칠해줘야 구역이 구분된다.
         
        # play time을 담기위한 변수 선언
        self.duration = ''
 
        # signal, btn_OOO = QPushButton
        self.btn_play.clicked.connect(self.clickPlay)
        self.btn_stop.clicked.connect(self.clickStop)
        self.btn_pause.clicked.connect(self.clickPause)
        self.btn_judgment.clicked.connect(self.clickJudgment)

        # list = QListWidget, (vol, bar) = QSlider
        self.bar.sliderMoved.connect(self.barChanged) # time line을 조정할 시
 
    def clickPlay(self):
        self.mp.playMedia(0)
 
    def clickStop(self):
        self.mp.stopMedia()
 
    def clickPause(self):
        self.mp.pauseMedia()
 
    def clickJudgment(self):
        test = predict.Predict(self.video_path)
        test.start_predict()
        # predict.py 객체 생성 후 영상을 predict -> 영상 다시 띄우기
 
    def barChanged(self, pos): 
        print(pos) 
        self.mp.posMoveMedia(pos) # 타임라인 변경에 따른 영상 조정   
 
    def updateState(self, msg): # 언제 메서드가 실행 되는지 모르겠음
        self.state.setText(msg) # Playing, Stop, Pause 상태
 
    def updateBar(self, duration): # 언제 메서드가 실행 되는지 모르겠음
        self.bar.setRange(0,duration)    
        self.bar.setSingleStep(int(duration/10))
        self.bar.setPageStep(int(duration/10))
        self.bar.setTickInterval(int(duration/10))
        td = datetime.timedelta(milliseconds=duration)        
        stime = str(td)
        idx = stime.rfind('.')
        self.duration = stime[:idx]
 
    def updatePos(self, pos):
        self.bar.setValue(pos)
        td = datetime.timedelta(milliseconds=pos)
        stime = str(td)
        idx = stime.rfind('.')
        stime = f'{stime[:idx]} / {self.duration}'
        self.playtime.setText(stime)
 
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wg = CWidget()
        self.setCentralWidget(self.wg)
        self.initUI()
    
    def initUI(self):
        # Menu Bar
        open_video_action = QAction(QIcon('./icon/open.png'), 'openVideo', self)
        open_video_action.triggered.connect(self.openVideo)
        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.addAction(open_video_action)

        # # QLable Icon
        # self.pixmap = QPixmap('./icon/online.png')
        # self.wg.label_2.setPixmap(self.pixmap) # 이미지 세팅
        # self.wg.label_2.resize(self.pixmap.width(), self.pixmap.height())


        self.setGeometry(100, 100, 1100, 600)
        self.show()

    def openVideo(self):
        files, ext = QFileDialog.getOpenFileNames(self
                                             , 'Select one or more files to open'
                                             , ''
                                             , 'Video (*.mp4 *.mpg *.mpeg *.avi *.wma)') 
        
        self.wg.video_path = files[0] 
        self.wg.mp.addMedia(files)       
 
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# from PyQt5.QtCore import Qt, QUrl
# from PyQt5.QtGui import QPalette
from PyQt5.uic import loadUi
from GUI_media import CMultiMedia # Qt에서 제공하는 멀티미디어 라이브러리 사용. PyQt5나 PySide2 등을 통해 사용 가능
import sys
import datetime

from models import predict


class CWidget(QWidget):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self) # Qt Designer로 생성한 ui파일 load
 
        # Multimedia Object, view = QVideoWidget
        self.mp = CMultiMedia(self, self.view) # media.py에 선언된 CMultiMedia 객체 생성...
        self.video_path = None

        # video background color
        pal = QPalette() # 각 위젯 상태에 대한 색상 그룹을 포함한다.
        pal.setColor(QPalette.Background, Qt.black) # (cr, color)
        self.view.setAutoFillBackground(True) # 
        self.view.setPalette(pal) # 검은색으로 칠해줘야 구역이 구분된다.
         
        # play time을 담기위한 변수 선언
        self.duration = ''
 
        # signal, btn_OOO = QPushButton
        self.btn_play.clicked.connect(self.clickPlay)
        self.btn_stop.clicked.connect(self.clickStop)
        self.btn_pause.clicked.connect(self.clickPause)
        self.btn_judgment.clicked.connect(self.clickJudgment)

        # list = QListWidget, (vol, bar) = QSlider
        self.bar.sliderMoved.connect(self.barChanged) # time line을 조정할 시
 
    def clickPlay(self):
        self.mp.playMedia(0)
 
    def clickStop(self):
        self.mp.stopMedia()
 
    def clickPause(self):
        self.mp.pauseMedia()
 
    def clickJudgment(self):
        test = predict.Predict(self.video_path)
        test.start_predict()
        # predict.py 객체 생성 후 영상을 predict -> 영상 다시 띄우기
 
    def barChanged(self, pos): 
        print(pos) 
        self.mp.posMoveMedia(pos) # 타임라인 변경에 따른 영상 조정   
 
    def updateState(self, msg): # 언제 메서드가 실행 되는지 모르겠음
        self.state.setText(msg) # Playing, Stop, Pause 상태
 
    def updateBar(self, duration): # 언제 메서드가 실행 되는지 모르겠음
        self.bar.setRange(0,duration)    
        self.bar.setSingleStep(int(duration/10))
        self.bar.setPageStep(int(duration/10))
        self.bar.setTickInterval(int(duration/10))
        td = datetime.timedelta(milliseconds=duration)        
        stime = str(td)
        idx = stime.rfind('.')
        self.duration = stime[:idx]
 
    def updatePos(self, pos):
        self.bar.setValue(pos)
        td = datetime.timedelta(milliseconds=pos)
        stime = str(td)
        idx = stime.rfind('.')
        stime = f'{stime[:idx]} / {self.duration}'
        self.playtime.setText(stime)
 
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wg = CWidget()
        self.setCentralWidget(self.wg)
        self.initUI()
    
    def initUI(self):
        # Menu Bar
        open_video_action = QAction(QIcon('./icon/open.png'), 'openVideo', self)
        open_video_action.triggered.connect(self.openVideo)
        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.addAction(open_video_action)

        # # QLable Icon
        # self.pixmap = QPixmap('./icon/online.png')
        # self.wg.label_2.setPixmap(self.pixmap) # 이미지 세팅
        # self.wg.label_2.resize(self.pixmap.width(), self.pixmap.height())


        self.setGeometry(100, 100, 1100, 600)
        self.show()

    def openVideo(self):
        files, ext = QFileDialog.getOpenFileNames(self
                                             , 'Select one or more files to open'
                                             , ''
                                             , 'Video (*.mp4 *.mpg *.mpeg *.avi *.wma)') 
        
        self.wg.video_path = files[0] 
        self.wg.mp.addMedia(files)       
 
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
