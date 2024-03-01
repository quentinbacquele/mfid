import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtCore import Qt
import os

class LauncherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.apply_dark_theme()
        self.setWindowTitle('App Launcher')
        self.resize(600, 500)
        
        layout = QVBoxLayout()
        
        # Load and display the image
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, '..', '..', 'images', 'main.jpg')
        imageLabel = QLabel(self)
        pixmap = QPixmap(image_path)
        imageLabel.setPixmap(pixmap)
        layout.addWidget(imageLabel)
        
        # Button to launch Body App
        bodyAppButton = self.create_button('Launch Body App', self.openBodyApp)
        layout.addWidget(bodyAppButton)

        # Button to launch Face App
        faceAppButton = self.create_button('Launch Face App', self.openFaceApp)
        layout.addWidget(faceAppButton)

        # Button to launch Identity App
        identityAppButton = self.create_button('Launch Identity App', self.openIdentityApp)
        layout.addWidget(identityAppButton)
        
        self.setLayout(layout)
    
    def apply_dark_theme(self):
        # Set dark theme colors
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)
        self.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                           "QPushButton:hover { background-color: #5A5A5A; }")

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        return button
    
    def openFaceApp(self):
        from mfid.face.face_app import FaceApp
        self.faceAppWindow = FaceApp()
        self.faceAppWindow.show()

    def openBodyApp(self):
        from mfid.body.body_app import BodyApp
        self.bodyAppWindow = BodyApp()
        self.bodyAppWindow.show()
        
    def openIdentityApp(self):
        from mfid.identity.identity_app import IdentityApp
        self.identityAppWindow = IdentityApp()
        self.identityAppWindow.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LauncherApp()
    ex.show()
    sys.exit(app.exec_())
