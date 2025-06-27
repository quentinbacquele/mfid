import os
import threading
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                           QFileDialog, QMessageBox, QLineEdit, QCheckBox, 
                           QLabel, QProgressBar, QComboBox, QHBoxLayout, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit
from mfid.utils.logging_utils import get_logger
from mfid.utils.notifications import Notifications, ProgressNotification
from mfid.utils.config_manager import ConfigManager
from mfid.utils.batch import BatchProcessor
from mfid.face.quality_assessment import assess_face_quality, filter_quality_faces

# Initialize logger
logger = get_logger('face_detector')

class DetectionWindow(QWidget):
    # Signal for batch progress updates
    progress_updated = pyqtSignal(int, int, object)
    processing_error = pyqtSignal(str, str)
    
    def __init__(self, saveFolder=None):
        super().__init__()
        self.saveFolder = saveFolder
        self.fileNames = []
        self.config = ConfigManager()
        self.batch_processor = BatchProcessor()
        
        # Connect batch processor signals
        self.batch_processor.progress_updated.connect(self.progress_updated)
        self.batch_processor.processing_error.connect(self.processing_error)
        self.batch_processor.processing_completed.connect(self.on_processing_completed)
        
        # Connect our signals
        self.progress_updated.connect(self.update_progress)
        self.processing_error.connect(self.show_error)
        
        self.initUI()

    def initUI(self):
        DarkTheme(self)  # Apply dark theme to the widget
        self.setWindowTitle('Face Detection')
        self.resize(700, 400)
        
        main_layout = QVBoxLayout()
        
        # File selection section
        file_selection_layout = QVBoxLayout()
        self.loadVideoButton = DarkButton('Load Videos/Images', self.openFileNameDialog)
        file_selection_layout.addWidget(self.loadVideoButton)
        
        # Selected files info
        self.files_label = QLabel("No files selected")
        self.files_label.setWordWrap(True)
        file_selection_layout.addWidget(self.files_label)
        
        # Options section
        options_layout = QVBoxLayout()
        options_label = QLabel("Detection Options")
        options_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        options_layout.addWidget(options_label)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["best_s_face.pt", "best_m_face.pt", "best_l_face.pt"])
        self.model_combo.setCurrentText("best_l_face.pt")  # Default to large model
        model_layout.addWidget(self.model_combo)
        options_layout.addLayout(model_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(50)  # 0.5 confidence
        self.conf_spin.setSuffix("%")
        conf_layout.addWidget(self.conf_spin)
        options_layout.addLayout(conf_layout)
        
        # Create checkboxes for options
        self.saveCoordsCheckbox = QCheckBox("Save Coordinates", self)
        self.saveCoordsCheckbox.setChecked(self.config.get('face_detection', 'save_coordinates', True))
        options_layout.addWidget(self.saveCoordsCheckbox)
        
        self.saveFramesCheckbox = QCheckBox("Save Full Frames", self)
        self.saveFramesCheckbox.setChecked(self.config.get('face_detection', 'save_full_frames', True))
        options_layout.addWidget(self.saveFramesCheckbox)
        
        self.qualityFilterCheckbox = QCheckBox("Filter by Face Quality", self)
        self.qualityFilterCheckbox.setChecked(False)
        options_layout.addWidget(self.qualityFilterCheckbox)
        
        # Frame skip
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Frame Skip:"))
        self.frameSkipSpin = QSpinBox()
        self.frameSkipSpin.setRange(1, 100)
        self.frameSkipSpin.setValue(self.config.get('face_detection', 'frame_skip', 5))
        skip_layout.addWidget(self.frameSkipSpin)
        options_layout.addLayout(skip_layout)
        
        # Progress section
        progress_layout = QVBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        progress_layout.addWidget(self.progressBar)
        
        self.statusLabel = QLabel('Status: Ready', self)
        progress_layout.addWidget(self.statusLabel)
        
        # Run button
        run_layout = QVBoxLayout()
        self.runButton = DarkButton('Run Detection', self.runDetection)
        run_layout.addWidget(self.runButton)
        
        # Add all sections to main layout
        main_layout.addLayout(file_selection_layout)
        main_layout.addLayout(options_layout)
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(run_layout)
        
        self.setLayout(main_layout)
        
     
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
                self.update_file_list()
        elif ret == 1:  # User chose Folder
            folder = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
            if folder:
                self.folderName = folder
                self.fileNames = self.recursive_file_search(folder)
                self.update_file_list()
                
                # Save to recent directories
                self.config.add_recent_directory(folder)

    def update_file_list(self):
        """Update the files label with the list of selected files"""
        if len(self.fileNames) == 0:
            self.files_label.setText("No files selected")
            return
            
        if len(self.fileNames) <= 5:
            file_text = "\n".join([os.path.basename(f) for f in self.fileNames])
        else:
            file_text = "\n".join([os.path.basename(f) for f in self.fileNames[:5]])
            file_text += f"\n... and {len(self.fileNames) - 5} more files"
            
        self.files_label.setText(f"Selected {len(self.fileNames)} files:\n{file_text}")

    def recursive_file_search(self, folder):
        supported_formats = ('.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG')
        file_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(supported_formats):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    
    def runDetection(self):
        if not self.fileNames:
            Notifications.warning(self, 'Please select videos/images to process')
            return
            
        if not self.saveFolder:
            # Ask user to select save folder if not already set
            self.saveFolder = str(QFileDialog.getExistingDirectory(self, "Select Save Directory"))
            if not self.saveFolder:
                Notifications.warning(self, 'Please select a save folder')
                return
        
        # Save settings
        self.config.set('face_detection', 'save_coordinates', self.saveCoordsCheckbox.isChecked())
        self.config.set('face_detection', 'save_full_frames', self.saveFramesCheckbox.isChecked())
        self.config.set('face_detection', 'frame_skip', self.frameSkipSpin.value())
        
        # Reset progress
        self.progressBar.setValue(0)
        self.statusLabel.setText('Status: Starting detection...')
        
        # Process each file
        total_files = len(self.fileNames)
        
        # Prepare batch processing
        self.batch_processor.clear_tasks()
        
        for file_path in self.fileNames:
            self.batch_processor.add_task(
                self.process_file, 
                file_path, 
                self.saveFolder,
                model=self.model_combo.currentText(),
                confidence=self.conf_spin.value() / 100.0,
                save_coords=self.saveCoordsCheckbox.isChecked(),
                save_frames=self.saveFramesCheckbox.isChecked(),
                skip_frames=self.frameSkipSpin.value(),
                quality_filter=self.qualityFilterCheckbox.isChecked()
            )
        
        # Show progress notification
        self.progress_notification = ProgressNotification.show_progress(self, f"Processing {total_files} files...")
        
        # Start batch processing
        self.batch_processor.process()
    
    def process_file(self, file_path, save_folder, **kwargs):
        """Process a single file"""
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Get parameters
            model_name = kwargs.get('model', 'best_l_face.pt')
            confidence = kwargs.get('confidence', 0.5)
            save_coords = kwargs.get('save_coords', True)
            save_frames = kwargs.get('save_frames', True)
            skip_frames = kwargs.get('skip_frames', 5)
            quality_filter = kwargs.get('quality_filter', False)
            
            # Load model
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            path_to_yolo_model = os.path.join(current_script_dir, '..', 'models', model_name)
            
            if not os.path.exists(path_to_yolo_model):
                logger.error(f"Model file not found: {path_to_yolo_model}")
                return {
                    'success': False, 
                    'file_path': file_path, 
                    'error': f"Model file not found: {model_name}"
                }
            
            model = YOLO(path_to_yolo_model)
            
            # Determine if file is image or video based on extension
            video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Create output directory
            os.makedirs(save_folder, exist_ok=True)
            
            # Process file based on type
            if file_ext in video_extensions:
                return self.process_video(
                    model, file_path, save_folder, confidence, save_coords, 
                    save_frames, skip_frames, quality_filter
                )
            elif file_ext in image_extensions:
                return self.process_image(
                    model, file_path, save_folder, confidence, save_coords, 
                    save_frames, quality_filter
                )
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return {
                    'success': False, 
                    'file_path': file_path, 
                    'error': f"Unsupported file type: {file_ext}"
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return {
                'success': False, 
                'file_path': file_path, 
                'error': str(e)
            }
    
    def process_video(self, model, video_path, save_folder, confidence, 
                     save_coords, save_frames, skip_frames, quality_filter):
        """Process a video file for face detection"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        face_count = 0
        frame_count = 0
        processed_frames = 0
        saved_faces = 0
        
        try:
            # Run detection using YOLO model
            results = model(video_path, conf=confidence, stream=True)
            
            for frame_idx, result in enumerate(results):
                if frame_idx % skip_frames != 0:
                    continue
                    
                processed_frames += 1
                frame = result.orig_img
                frame_faces = []
                
                # Process detections in this frame
                if len(result.boxes) > 0:
                    # Create coordinates file if needed
                    if save_coords:
                        coords_file_path = os.path.join(save_folder, f'{video_name}_frame_{frame_count}.txt')
                        coords_file = open(coords_file_path, 'w')
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        face_img = frame[y1:y2, x1:x2].copy()
                        
                        # Quality assessment if enabled
                        if quality_filter:
                            quality_result = assess_face_quality(face_img)
                            if not quality_result['is_good_quality']:
                                continue
                        
                        # Save coordinates if requested
                        if save_coords:
                            coords_file.write(f"{x1},{y1},{x2},{y2}\n")
                        
                        # Save face crop
                        face_path = os.path.join(save_folder, f'{video_name}_face_{face_count}_{frame_count}.jpg')
                        cv2.imwrite(face_path, face_img)
                        saved_faces += 1
                        
                        # Draw rectangle for visualization
                        if save_frames:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        face_count += 1
                    
                    # Close coordinates file if opened
                    if save_coords:
                        coords_file.close()
                    
                    # Save frame with detections if requested
                    if save_frames:
                        frame_path = os.path.join(save_folder, f'{video_name}_frame_{frame_count}.jpg')
                        cv2.imwrite(frame_path, frame)
                
                frame_count += 1
            
            return {
                'success': True,
                'file_path': video_path,
                'file_name': video_name,
                'face_count': saved_faces,
                'processed_frames': processed_frames,
                'type': 'video'
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'file_path': video_path,
                'error': str(e),
                'type': 'video'
            }
    
    def process_image(self, model, image_path, save_folder, confidence, 
                     save_coords, save_frames, quality_filter):
        """Process an image file for face detection"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        face_count = 0
        saved_faces = 0
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'file_path': image_path,
                    'error': 'Failed to load image',
                    'type': 'image'
                }
            
            # Run detection
            result = model(image, conf=confidence)[0]
            
            # Create coordinates file if needed
            if save_coords and len(result.boxes) > 0:
                coords_file_path = os.path.join(save_folder, f'{image_name}_coords.txt')
                coords_file = open(coords_file_path, 'w')
            
            # Process detections
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_img = image[y1:y2, x1:x2].copy()
                
                # Quality assessment if enabled
                if quality_filter:
                    quality_result = assess_face_quality(face_img)
                    if not quality_result['is_good_quality']:
                        continue
                
                # Save coordinates if requested
                if save_coords:
                    coords_file.write(f"{x1},{y1},{x2},{y2}\n")
                
                # Save face crop
                face_path = os.path.join(save_folder, f'{image_name}_face_{face_count}.jpg')
                cv2.imwrite(face_path, face_img)
                saved_faces += 1
                
                # Draw rectangle for visualization
                if save_frames:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                face_count += 1
            
            # Close coordinates file if opened
            if save_coords and len(result.boxes) > 0:
                coords_file.close()
            
            # Save frame with detections if requested
            if save_frames and face_count > 0:
                frame_path = os.path.join(save_folder, f'{image_name}_annotated.jpg')
                cv2.imwrite(frame_path, image)
            
            return {
                'success': True,
                'file_path': image_path,
                'file_name': image_name,
                'face_count': saved_faces,
                'type': 'image'
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'file_path': image_path,
                'error': str(e),
                'type': 'image'
            }
    
    @pyqtSlot(int, int, object)
    def update_progress(self, current, total, result):
        """Update progress UI with batch processing progress"""
        percentage = int(current * 100 / total)
        self.progressBar.setValue(percentage)
        
        if result and 'success' in result:
            if result['success']:
                if result['type'] == 'video':
                    status_text = f"Processed: {os.path.basename(result['file_path'])} - Found {result['face_count']} faces in {result['processed_frames']} frames"
                else:
                    status_text = f"Processed: {os.path.basename(result['file_path'])} - Found {result['face_count']} faces"
            else:
                status_text = f"Error processing {os.path.basename(result['file_path'])}: {result.get('error', 'Unknown error')}"
                
            self.statusLabel.setText(status_text)
        
        # Update progress notification
        if self.progress_notification:
            if result and 'success' in result:
                if result['success']:
                    self.progress_notification.update_progress(percentage, f"Processing file {current} of {total}")
                else:
                    self.progress_notification.update_progress(percentage, f"Error in file {current} of {total}")
            else:
                self.progress_notification.update_progress(percentage, f"Processing file {current} of {total}")
    
    @pyqtSlot(str, str)
    def show_error(self, error_msg, file_path):
        """Show error notification"""
        Notifications.error(self, f"Error processing {os.path.basename(file_path)}: {error_msg}")
    
    @pyqtSlot()
    def on_processing_completed(self):
        """Handle batch processing completion"""
        logger.info("Face detection completed")
        
        results = self.batch_processor.get_results()
        total_faces = sum(result.get('face_count', 0) for result in results if result.get('success', False))
        success_count = sum(1 for result in results if result.get('success', False))
        error_count = len(results) - success_count
        
        self.statusLabel.setText(f"Detection completed. Processed {len(results)} files, found {total_faces} faces. {error_count} errors.")
        
        # Complete progress notification
        if self.progress_notification:
            if error_count > 0:
                self.progress_notification.complete(f"Completed with {error_count} errors. Found {total_faces} faces.")
            else:
                self.progress_notification.complete(f"Detection completed successfully. Found {total_faces} faces.")
        
        # Show completion notification
        if error_count > 0:
            Notifications.warning(self, f"Detection completed with {error_count} errors. Found {total_faces} faces.")
        else:
            Notifications.success(self, f"Detection completed successfully. Found {total_faces} faces.")
