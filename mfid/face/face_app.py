import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
from mfid.face.face_detector import DetectionWindow
from mfid.face.annotation_interface import AnnotationWindow
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit

class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        DarkTheme(self) 
        self.setWindowTitle('YOLOv8 Face Detection')
        self.resize(700, 100)
        
        layout = QVBoxLayout()
        
        # Main Menu Buttons
        self.selectSaveFolderButton = DarkButton('Select Save Folder', self.openFolderDialog)
        self.detectionButton = DarkButton('Detection', self.openDetectionWindow)
        self.annotationButton = DarkButton('Annotation', self.openAnnotationWindow)
        
        for widget in [self.selectSaveFolderButton, self.detectionButton, self.annotationButton]:
            layout.addWidget(widget)
        
        self.setLayout(layout)

    def openFolderDialog(self):
        self.saveFolder = str(QFileDialog.getExistingDirectory(self, "Select Save Directory"))
        if self.saveFolder:
            self.statusLabel.setText(f'Save Folder: {self.saveFolder}')

    def openDetectionWindow(self):
        self.detectionWindow = DetectionWindow()
        self.detectionWindow.show()

    def openAnnotationWindow(self):
        self.annotationWindow = AnnotationWindow()
        self.annotationWindow.show()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceApp()
    ex.show()
    sys.exit(app.exec_())