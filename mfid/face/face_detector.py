import os
import threading
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLineEdit, QCheckBox, QLabel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit



class DetectionWindow(QWidget):
    def __init__(self, saveFolder=None):
        super().__init__()
        self.saveFolder = saveFolder
        self.initUI()

    def initUI(self):
        DarkTheme(self)  # Apply dark theme to the widget
        self.setWindowTitle('Detection')
        self.resize(500, 300)
        
        layout = QVBoxLayout()
        
        # Correctly use DarkButton and DarkLineEdit as standalone functions
        self.loadVideoButton = DarkButton('Load Video/Folder', self.openFileNameDialog)
        self.saveCoordsCheckbox = QCheckBox("Save Coordinates", self)
        self.saveFramesCheckbox = QCheckBox("Save Full Frames", self)
        self.saveFramesCheckbox.setChecked(True)
        self.frameSkipInput = QLineEdit(self)
        DarkLineEdit(self.frameSkipInput, 'Enter frame skip value (e.g., 5)')  # Apply dark line edit style
        self.runButton = DarkButton('Run Detection', self.runDetection)
        self.statusLabel = QLabel('Status: Waiting for input', self)
        self.progressLabel = QLabel('Progress: 0 / 0', self)
        
        # Add all widgets to the layout
        layout.addWidget(self.loadVideoButton)
        layout.addWidget(self.saveCoordsCheckbox)
        layout.addWidget(self.saveFramesCheckbox)
        layout.addWidget(self.frameSkipInput)
        layout.addWidget(self.runButton)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.progressLabel)
        
        self.setLayout(layout)
        
     
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
            path_to_yolo_model = os.path.join(current_script_dir, '..', 'models/best_l_face.pt')
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
