from PyQt5.QtMultimedia import QMediaPlaylist, QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QObject, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
 
class CMultiMedia(QObject):
 
    state_signal = pyqtSignal(str)
    duration_signal = pyqtSignal(int)
    position_signal = pyqtSignal(int)
 
    def __init__(self, widget, video_widget):  
        super().__init__()
        self.parent = widget
        self.player = QMediaPlayer(widget, flags=QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(video_widget)
        self.list = QMediaPlaylist()
        self.player.setPlaylist(self.list)
 
        # signal
        self.player.error.connect(self.errorHandle) 
        self.player.stateChanged.connect(self.stateChanged)
        self.player.durationChanged.connect(self.durationChanged)
        self.player.positionChanged.connect(self.positionChanged)
 
        # user signal
        self.state_signal.connect(self.parent.updateState) # 시그널에 updateState 함수를 슬롯으로 추가
        self.duration_signal.connect(self.parent.updateBar)
        self.position_signal.connect(self.parent.updatePos)
 
    def addMedia(self, file):
        self.list.clear()

        url = QUrl.fromLocalFile(file)
        self.list.addMedia(QMediaContent(url))
 
    def playMedia(self, index):
        self.list.setCurrentIndex(index)
        self.player.play()
 
    def stopMedia(self):
        self.player.stop()
 
    def pauseMedia(self):
        self.player.pause()
 
    def posMoveMedia(self, pos):
        self.player.setPosition(pos)
 
    def stateChanged(self, state): # 발생시 위젯으로 전달
        msg = ''
        if state==QMediaPlayer.StoppedState:
            msg = 'Stopped'
        elif state==QMediaPlayer.PlayingState:
            msg = 'Playing'
        else:
            msg = 'Paused'
        self.state_signal.emit(msg)
 
    def durationChanged(self, duration):
        self.duration_signal.emit(duration)
 
    def positionChanged(self, pos):        
        self.position_signal.emit(pos)
 
    def errorHandle(self, e):
        self.state_signal.emit(self.player.errorString())
