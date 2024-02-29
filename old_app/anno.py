import sys
import os
import threading
import cv2
import shutil
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox, QLineEdit, QCheckBox
from PyQt5.QtGui import QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt


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
        self.apply_dark_theme()
        self.image_label = QLabel(self)
        self.text_input = QLineEdit(self)
        self.next_button = self.create_button('Next Image', self.processAnnotation)
        self.prev_button = self.create_button('Previous Image', self.showPreviousImage)
        self.showFrameButton = self.create_button('Show Full Frame', self.showFullFrame)
        self.falsePositiveButton = self.create_button('False Positive', self.handleFalsePositive)
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

    def apply_dark_theme(self):
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
        self.setStyleSheet("QLabel, QLineEdit, QCheckBox { color: white; }")

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                             "QPushButton:hover { background-color: #5A5A5A; }")
        return button


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


class YOLOv8App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.apply_dark_theme()
        self.setWindowTitle('YOLOv8 Face Detection')
        self.resize(1000, 200)
        
        layout = QVBoxLayout()
        
        # Main Menu Buttons
        self.selectSaveFolderButton = self.create_button('Select Save Folder', self.openFolderDialog)
        self.detectionButton = self.create_button('Detection', self.openDetectionWindow)
        self.annotationButton = self.create_button('Annotation', self.openAnnotationWindow)
        
        for widget in [self.selectSaveFolderButton, self.detectionButton, self.annotationButton]:
            layout.addWidget(widget)
        
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
        self.setStyleSheet("QLabel, QCheckBox { color: white; }")

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                             "QPushButton:hover { background-color: #5A5A5A; }")
        return button

    def apply_line_edit_style(self, line_edit_widget, placeholder_text):
        line_edit_widget.setPlaceholderText(placeholder_text)
        # Ensure placeholder text is black for visibility
        line_edit_widget.setStyleSheet("QLineEdit { color: black; } QLineEdit::placeholder { color: black; }")

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

class DetectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.apply_dark_theme()
        self.setWindowTitle('Detection')
        self.resize(500, 300)
        
        layout = QVBoxLayout()
        
        self.loadVideoButton = self.create_button('Load Video/Folder', self.openFileNameDialog)
        self.saveCoordsCheckbox = QCheckBox("Save Coordinates", self)
        self.saveFramesCheckbox = QCheckBox("Save Full Frames", self)
        self.saveFramesCheckbox.setChecked(True)
        self.frameSkipInput = QLineEdit(self)
        self.apply_line_edit_style(self.frameSkipInput, 'Enter frame skip value (e.g., 5)')
        self.runButton = self.create_button('Run Detection', self.runDetection)
        
        # Add all widgets and set the layout
        layout.addWidget(self.loadVideoButton)
        layout.addWidget(self.saveCoordsCheckbox)
        layout.addWidget(self.saveFramesCheckbox)
        layout.addWidget(self.frameSkipInput)
        layout.addWidget(self.runButton)
        
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
        self.setStyleSheet("QLabel, QCheckBox { color: white; }")

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                             "QPushButton:hover { background-color: #5A5A5A; }")
        return button

    def apply_line_edit_style(self, line_edit_widget, placeholder_text):
        line_edit_widget.setPlaceholderText(placeholder_text)
        # Ensure placeholder text is black for visibility
        line_edit_widget.setStyleSheet("QLineEdit { color: black; } QLineEdit::placeholder { color: black; }")

        
    def openFileNameDialog(self):
        msgBox = QMessageBox()
        msgBox.setText("Select the input type")
        msgBox.addButton(QPushButton('Files'), QMessageBox.YesRole)
        msgBox.addButton(QPushButton('Folder'), QMessageBox.NoRole)
        ret = msgBox.exec_()

        if ret == 0:  # User chose Files
            options = QFileDialog.Options()
            files, _ = QFileDialog.getOpenFileNames(self, "Select Videos/Images", "", "All Files (*);;Video Files (*.mp4 *.avi *.mov *.MOV *.MP4 *.AVI);;Image Files (*.jpg *.jpeg *.png *.JPG *.JPEG)", options=options)
            if files:
                self.fileNames = files
                self.statusLabel.setText('Files Selected: ' + ', '.join([os.path.basename(file) for file in files]))
        elif ret == 1:  # User chose Folder
            folder = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
            if folder:
                self.folderName = folder
                self.fileNames = self.recursive_file_search(folder)
                self.statusLabel.setText('Folder Selected: ' + folder)

    def recursive_file_search(self, folder):
        supported_formats = ('.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG')
        file_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(supported_formats):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    
    def runDetection(self):
        if not hasattr(self, 'fileNames') or not hasattr(self, 'saveFolder'):
            QMessageBox.warning(self, 'Missing Information', 'Please select videos and a save folder.')
            return

        threading.Thread(target=self.processDetection, args=(self.fileNames, self.saveFolder), daemon=True).start()
    
    def processDetection(self,  file_paths, save_folder):
        # Update status
        self.statusLabel.setText('Running detection...')
        total_videos = len(file_paths)
        processed_videos = 0
 
        def process_video(video_path, save_folder, skip_frames=5):
            nonlocal processed_videos
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            path_to_yolo_model = os.path.join(current_script_dir, 'models/best_l_face.pt')
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            model = YOLO(path_to_yolo_model)
            results = model(video_path, stream=True)

            os.makedirs(save_folder, exist_ok=True)
            face_counter = 0
            frame_counter = 0

            for result in results:
                if frame_counter % skip_frames == 0:  # Process only every Nth frame
                    frame = result.orig_img
                    if self.saveCoordsCheckbox.isChecked():
                        with open(os.path.join(save_folder, f'{video_name}_frame_{frame_counter}.txt'), 'w') as file:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                file.write(f"{x1},{y1},{x2},{y2}\n")
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cropped_img = frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(save_folder, f'{video_name}_face_{face_counter}_{frame_counter}.jpg'), cropped_img)
                        if self.saveFramesCheckbox.isChecked():  # Check if the user wants to save full frames
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.imwrite(os.path.join(save_folder, f'{video_name}_frame_{frame_counter}.jpg'), frame)
                        face_counter += 1
                frame_counter += 1
            
            processed_videos += 1
            self.progressLabel.setText(f'Progress: {processed_videos} / {total_videos}')
            QApplication.processEvents()  # Ensure the GUI updates the label text

        try:
            skip_frames = int(self.frameSkipInput.text())  # Get frame skip value from input
        except ValueError:
            skip_frames = 1  # Default value if input is invalid or empty

        try:
            for file_path in file_paths:
                process_video(file_path, save_folder, skip_frames)
        except Exception as e:
            self.statusLabel.setText(f'Error during detection: {e}')
            return
        
        self.statusLabel.setText('Detection completed.')

class AnnotationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.apply_dark_theme()
        self.setWindowTitle('Annotation')
        self.resize(500, 300)
        
        layout = QVBoxLayout()
        
        self.loadFacesButton = self.create_button('Load Folder with Extracted Faces', self.openFacesFolderDialog)
        self.annotateButton = self.create_button('Annotate Cropped Faces', self.launchAnnotator)
        self.autoAnnotateInput = QLineEdit(self)
        self.apply_line_edit_style(self.autoAnnotateInput, 'Enter keyword for automatic annotation (e.g., kabuki)')
        self.autoAnnotateButton = self.create_button('Automatic Annotate', self.autoAnnotate)
        self.deleteFramesButton = self.create_button('Delete Full Frames', self.deleteFullFrames)
        
        # Add all widgets and set the layout
        layout.addWidget(self.loadFacesButton)
        layout.addWidget(self.annotateButton)
        layout.addWidget(self.autoAnnotateInput)
        layout.addWidget(self.autoAnnotateButton)
        layout.addWidget(self.deleteFramesButton)
        
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
        self.setStyleSheet("QLabel, QCheckBox { color: white; }")

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                             "QPushButton:hover { background-color: #5A5A5A; }")
        return button

    def apply_line_edit_style(self, line_edit_widget, placeholder_text):
        line_edit_widget.setPlaceholderText(placeholder_text)
        # Ensure placeholder text is black for visibility
        line_edit_widget.setStyleSheet("QLineEdit { color: black; } QLineEdit::placeholder { color: black; }")
    
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
            QMessageBox.warning(self, 'Missing Information', 'Please run detection first to generate cropped faces or select a folder with faces to annotate.')
    
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

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOv8App()
    ex.show()
    sys.exit(app.exec_())