import os
import shutil
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLineEdit, QLabel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
from mfid.face.face_annotator import ImageAnnotator
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit

class AnnotationWindow(QWidget):
    def __init__(self, saveFolder=None):
        super().__init__()
        self.saveFolder = saveFolder
        self.initUI()

    def initUI(self):
        DarkTheme(self) 
        self.setWindowTitle('Annotation')
        self.resize(500, 300)
        
        layout = QVBoxLayout()
        
        self.loadFacesButton = DarkButton('Load Folder with Extracted Faces', self.openFacesFolderDialog)
        self.annotateButton = DarkButton('Annotate Cropped Faces', self.launchAnnotator)
        self.autoAnnotateInput = QLineEdit(self)
        DarkLineEdit(self.autoAnnotateInput, 'Enter keyword for automatic annotation (e.g., kabuki)')
        self.autoAnnotateButton = DarkButton('Automatic Annotate', self.autoAnnotate)
        self.deleteFramesButton = DarkButton('Delete Full Frames', self.deleteFullFrames)
        self.statusLabel = QLabel('No folder selected', self)  
        self.statusLabel.setWordWrap(True)
        
        # Add all widgets and set the layout
        layout.addWidget(self.loadFacesButton)
        layout.addWidget(self.annotateButton)
        layout.addWidget(self.autoAnnotateInput)
        layout.addWidget(self.autoAnnotateButton)
        layout.addWidget(self.deleteFramesButton)
        layout.addWidget(self.statusLabel)
        
        self.setLayout(layout)
        
    def openFacesFolderDialog(self):
        self.facesFolder = str(QFileDialog.getExistingDirectory(self, "Select Folder with Extracted Faces"))
        if self.facesFolder:
            self.statusLabel.setText(f'Faces Folder: {self.facesFolder}')
    
    def deleteFullFrames(self):
        if hasattr(self, 'saveFolder'):
            for img in os.listdir(self.saveFolder):
                if 'frame_' in img:
                    os.remove(os.path.join(self.saveFolder, img))
            QMessageBox.information(self, 'Deletion Complete', 'All full frames have been deleted.')
        else:
            QMessageBox.warning(self, 'Folder Not Set', 'Please select a folder first.')
    
    def launchAnnotator(self):
        if hasattr(self, 'saveFolder'):
            self.annotator = ImageAnnotator(self.saveFolder)
            self.annotator.show()
        elif hasattr(self, 'facesFolder'):
            self.annotator = ImageAnnotator(self.facesFolder)
            self.annotator.show()
        else:
            QMessageBox.warning(self, 'Missing Information', 'Please select a folder with faces to annotate.')
    
    def autoAnnotate(self):
        keyword = self.autoAnnotateInput.text().strip().lower()
        if not keyword:
            QMessageBox.warning(self, 'No Keyword', 'Please enter a keyword for automatic annotation.')
            return

        if not hasattr(self, 'facesFolder') or not os.path.exists(self.facesFolder):
            QMessageBox.warning(self, 'No Folder Selected', 'Please select a folder with extracted faces first.')
            return

        target_folder = os.path.join(self.facesFolder, keyword)
        os.makedirs(target_folder, exist_ok=True)

        for img in os.listdir(self.facesFolder):
            if 'frame_' in img.lower():
                continue  # Skip images with 'frame' in the name

            if keyword in img.lower():
                source_path = os.path.join(self.facesFolder, img)
                shutil.move(source_path, target_folder)

        QMessageBox.information(self, 'Automatic Annotation Complete', f'Images with keyword "{keyword}" have been moved to the folder: {target_folder}')