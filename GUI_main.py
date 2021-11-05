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
import pandas as pd


class CWidget(QWidget):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self) # Qt Designer로 생성한 ui파일 load
 
        # Multimedia Object, view = QVideoWidget
        self.mp = CMultiMedia(self, self.view) # media.py에 선언된 CMultiMedia 객체 생성...
        self.original_video_path = None

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
        self.list.itemDoubleClicked.connect(self.dbClickLogs)
        self.bar.sliderMoved.connect(self.barChanged) # time line을 조정할 시
 
    def clickPlay(self):
        self.mp.playMedia(0)
 
    def clickStop(self):
        self.mp.stopMedia()
 
    def clickPause(self):
        self.mp.pauseMedia()
 
    def clickJudgment(self):
        test = predict.Predict(self.original_video_path)
        output_video_path, log_path, count_path, self.video_fps  = test.start_predict() # return = output_video_path, log file, count file
        
        # output_video_path를 이용
        self.mp.addMedia(output_video_path)

        # log_file를 이용
        log_csv = pd.read_csv(log_path)
        for i in range(len(log_csv)):
            if log_csv.iloc[i][2] == 2:
                log = '★★★ ' + str(log_csv.iloc[i][1]) + ' ' + 'A 행동' + ' ★★★'
                self.list.addItem(log)
            elif log_csv.iloc[i][2] == 3:
                log = '★★★ ' + str(log_csv.iloc[i][1]) + ' ' + 'B 행동' + ' ★★★'
                self.list.addItem(log)
            else:
                log = '★★★ ' + str(log_csv.iloc[i][1]) + ' ' + 'C 행동' + ' ★★★'
                self.list.addItem(log)

        # count_path를 이용
        count_csv = pd.read_csv(count_path)

        count_A = count_csv.iloc[0][1]
        count_B = count_csv.iloc[0][2]
        count_C = count_csv.iloc[0][3]

        count_total = count_A + count_B + count_C

        self.label_total.setText(f'abnormal - {count_total}')
        self.label_total.adjustSize()
        self.label_A.setText(f'A 행동 - {count_A}')
        self.label_total.adjustSize()
        self.label_B.setText(f'B 행동 - {count_B}')
        # self.label_B.adjustSize()
        # self.label_C.setText(f'C 행동 - {count_C}')
        # self.label_C.adjustSize()


        # self.wg.lbl_pos.setText(txt)
        # self.wg.lbl_pos.adjustSize()  # 내용에 맞게 위젯의 크기를 조정한다. https://doc.qt.io/qt-5/qwidget.html#adjustSize
        

        #self.mp.addMedia(output_video_path) # clickPlay 하면 실행됨
 
    def dbClickLogs(self):
        row = self.list.currentRow()
        row_val= self.list.item(row)
        row_val = row_val.text() 

        time_val = row_val.split()[1]
        start_t = time_val.split('-')[0]
        end_t = time_val.split('-')[1]

        mm = (int(start_t) / self.video_fps) * 1000

        self.mp.moveStart_t(mm)

    def barChanged(self, pos): 
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
        
        self.wg.original_video_path = files[0] 
        self.wg.mp.addMedia(files[0])       
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
