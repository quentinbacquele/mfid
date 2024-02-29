import os
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox, QLineEdit, QCheckBox
from PyQt5.QtGui import QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit



class ImageAnnotator(QWidget):
    def __init__(self, folderPath):
        super().__init__()
        self.folderPath = folderPath
        self.images = sorted([img for img in os.listdir(folderPath) if img.endswith('.jpg')])
        self.currentIndex = 0
        self.currentVideo = None
        self.annotations = {}  # Dictionary to store annotations
        self.initUI()

    def initUI(self):
        DarkTheme(self) 
        self.image_label = QLabel(self)
        self.text_input = QLineEdit(self)
        self.next_button = DarkButton('Next Image', self.processAnnotation)
        self.prev_button = DarkButton('Previous Image', self.showPreviousImage)
        self.showFrameButton = DarkButton('Show Full Frame', self.showFullFrame)
        self.falsePositiveButton = DarkButton('False Positive', self.handleFalsePositive)
        self.retainLabelCheckbox = QCheckBox("Retain Label for Next Image", self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_input)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.showFrameButton)
        layout.addWidget(self.falsePositiveButton)
        layout.addWidget(self.retainLabelCheckbox)
        
        self.setLayout(layout)
        self.setWindowTitle('Image Annotator')
        self.showNextImage()



    def processAnnotation(self):
        annotation = self.text_input.text().strip()
        if annotation:
            self.annotations[self.images[self.currentIndex]] = annotation

        self.currentIndex += 1
        self.showNextImage()

    def showPreviousImage(self):
        if self.currentIndex > 0:
            self.currentIndex -= 1
            self.showNextImage()

    def showFullFrame(self):
        try:
            face_image_name = self.images[self.currentIndex]
            parts = face_image_name.split('_')
            frame_number = parts[-1].split('.')[0]  # Extracting the frame number from the end
            video_name = '_'.join(parts[:-3])  # Joining all parts except the last three (face, face_counter, frame_counter)

            frame_path = os.path.join(self.folderPath, f'{video_name}_frame_{frame_number}.jpg')
            print(f"Attempting to open full frame: {frame_path}")  # Debugging print

            if os.path.exists(frame_path):
                pixmap = QPixmap(frame_path)
                if pixmap.isNull():
                    raise Exception("Failed to load image.")

                self.fullFrameWindow = QLabel()
                self.fullFrameWindow.setPixmap(pixmap.scaled(1200, 1200, Qt.KeepAspectRatio))
                self.fullFrameWindow.show()
            else:
                print("Frame path does not exist.")  # Debugging print
        except Exception as e:
            print(f"Error in showFullFrame: {e}")  # Error handling


    def showNextImage(self):
        # Skip full frames and show only cropped face images
        while self.currentIndex < len(self.images) and 'frame_' in self.images[self.currentIndex]:
            self.currentIndex += 1

        if self.currentIndex < len(self.images):
            current_image = self.images[self.currentIndex]
            parts = current_image.split('_')
            # Joining all parts except the last three (face, face_counter, frame_counter)
            video_name = '_'.join(parts[:-3])

            if self.currentVideo != video_name:
                self.currentVideo = video_name
                self.setWindowTitle(f'Image Annotator - {video_name}')
                self.retainLabelCheckbox.setChecked(False)

            pixmap = QPixmap(os.path.join(self.folderPath, current_image))
            self.image_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio))

            if current_image in self.annotations:
                self.text_input.setText(self.annotations[current_image])
            elif not self.retainLabelCheckbox.isChecked():
                self.text_input.clear()
        else:
            QMessageBox.information(self, 'Done', 'All images have been annotated.')
            self.close()


    def handleFalsePositive(self):
        os.remove(os.path.join(self.folderPath, self.images[self.currentIndex]))
        self.currentIndex += 1
        self.showNextImage()

    def finalizeAnnotations(self):
        for image, annotation in self.annotations.items():
            annotation_folder = os.path.join(self.folderPath, annotation)
            os.makedirs(annotation_folder, exist_ok=True)

            source_path = os.path.join(self.folderPath, image)
            if os.path.exists(source_path):
                existing_files = len([name for name in os.listdir(annotation_folder) if os.path.isfile(os.path.join(annotation_folder, name))])
                new_filename = f"{annotation}{existing_files + 1}.jpg"
                os.rename(source_path, os.path.join(annotation_folder, new_filename))

    def closeEvent(self, event):
        self.finalizeAnnotations()
        #self.deleteFullFrames()
        super().closeEvent(event)