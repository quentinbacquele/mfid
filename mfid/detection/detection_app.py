import sys
import os
import shutil
import multiprocessing
import threading
import platform
import torch
import cv2
import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QFormLayout, QMessageBox, QDoubleSpinBox, QComboBox, QCheckBox, 
                           QHBoxLayout, QSpinBox, QProgressBar, QGroupBox, QRadioButton, QButtonGroup, 
                           QScrollArea, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QEvent
from PyQt5.QtGui import QPalette, QColor, QFont
from ultralytics import YOLO
from mfid.utils.theme_dark import DarkTheme, DarkButton
from mfid.utils.notifications import Notifications, ProgressNotification
from mfid.utils.config_manager import ConfigManager
from mfid.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger('detection_app') # Changed logger name

# Event filter to ignore scroll wheel events on specific widgets
class ScrollEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)

# Worker class to handle processing in a separate thread
class DetectionWorker(QObject):
    progress_updated = pyqtSignal(int, int, str)
    detection_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def process_files(self, files, params, output_path, model_base_path, selected_model_file):
        try:
            # Construct model path
            model_path = os.path.join(model_base_path, selected_model_file)
            
            logger.info(f"Loading model: {model_path}")
            if not os.path.exists(model_path):
                self.error_occurred.emit(f"Model file not found: {model_path}")
                return
            model = YOLO(model_path)
            
            save_summary = params.pop('save_summary', True)
            save_videos_images = params.pop('save_videos_images', False)
            save_txt = params.pop('save_txt', False)
            save_crops = params.pop('save_crops', False)
            sort_detections = params.pop('sort_detections', False)
            frame_skip = params.pop('frame_skip', 0)

            detections_count = 0
            no_detections_count = 0
            
            # Create output directories
            os.makedirs(output_path, exist_ok=True)
            if save_crops:
                crops_dir = os.path.join(output_path, "crops")
                os.makedirs(crops_dir, exist_ok=True)
            if save_videos_images:
                annotated_dir = os.path.join(output_path, "annotated")
                os.makedirs(annotated_dir, exist_ok=True)
            if save_txt:
                coords_dir = os.path.join(output_path, "coordinates")
                os.makedirs(coords_dir, exist_ok=True)
            if sort_detections:
                positive_dir = os.path.join(output_path, "positive_detections")
                negative_dir = os.path.join(output_path, "negative_detections")
                os.makedirs(positive_dir, exist_ok=True)
                os.makedirs(negative_dir, exist_ok=True)
            
            summary_file_path = os.path.join(output_path, "detection_summary.txt")
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write("filename,has_detection,min_detections,max_detections\n")
                
                for i, file_path in enumerate(files):
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]
                    logger.info(f"Processing file {i+1}/{len(files)}: {file_name}")
                    
                    self.progress_updated.emit(i+1, len(files), file_name)
                    
                    # Ensure 'conf' and 'iou' are present for model.predict
                    predict_params = {
                        'conf': params.get('conf', 0.25),
                        'iou': params.get('iou', 0.45),
                        'device': params.get('device', None),
                        'imgsz': params.get('imgsz', 640),
                    }
                    # Filter out None device value
                    if predict_params['device'] is None:
                        del predict_params['device']

                    # Handle frame skipping for video files
                    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            logger.warning(f"Could not open video file: {file_path}")
                            summary_file.write(f"{file_name},0,0,0\n")
                            if sort_detections:
                                shutil.copy2(file_path, os.path.join(negative_dir, file_name))
                            no_detections_count += 1
                            continue

                        # Get video properties
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        # Create video writer if saving annotated video
                        if save_videos_images:
                            output_video_path = os.path.join(annotated_dir, f"{base_name}_annotated.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                        processed_frames_count = 0
                        frame_idx = 0
                        all_results_for_file = []
                        
                        # Create coordinates file if needed
                        if save_txt:
                            coords_file_path = os.path.join(coords_dir, f"{base_name}_coordinates.txt")
                            coords_file = open(coords_file_path, 'w')

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                                frame_idx += 1
                                continue

                            results = model.predict(frame, **predict_params)
                            result = results[0]  # Get first result
                            all_results_for_file.append(result)

                            if len(result.boxes) > 0 and save_txt:
                                # Save coordinates for this frame
                                coords_file.write(f"Frame {frame_idx}:\n")
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = result.names[cls]
                                    coords_file.write(f"{label} {conf:.4f} {x1} {y1} {x2} {y2}\n")

                            if save_videos_images:
                                # Draw boxes on frame
                                annotated_frame = result.plot()
                                out.write(annotated_frame)

                            processed_frames_count += 1
                            frame_idx += 1

                        if save_txt:
                            coords_file.close()
                        if save_videos_images:
                            out.release()
                        cap.release()
                        
                        logger.info(f"Processed {processed_frames_count} frames for {file_name}")
                        results = all_results_for_file
                    else:  # Process images
                        results = model.predict(file_path, **predict_params)
                        result = results[0]  # Get first result

                        if save_videos_images and len(result.boxes) > 0:
                            # Save annotated image
                            annotated_img = result.plot()
                            cv2.imwrite(os.path.join(annotated_dir, f"{base_name}_annotated.jpg"), annotated_img)

                        if save_txt and len(result.boxes) > 0:
                            # Save coordinates
                            coords_file_path = os.path.join(coords_dir, f"{base_name}_coordinates.txt")
                            with open(coords_file_path, 'w') as coords_file:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = result.names[cls]
                                    coords_file.write(f"{label} {conf:.4f} {x1} {y1} {x2} {y2}\n")
                    
                    has_detections = False
                    min_detections_in_frame = float('inf')
                    max_detections_in_frame = 0
                    detection_idx = 0
                    
                    for result in results:
                        boxes = result.boxes
                        num_detections_in_current_frame = len(boxes)
                        
                        if num_detections_in_current_frame > 0:
                            has_detections = True
                            
                            # Save cropped detections if enabled
                            if save_crops:
                                frame = result.orig_img
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = result.names[cls]
                                    
                                    # Extract the detection crop
                                    crop = frame[y1:y2, x1:x2].copy()
                                    
                                    # Save with informative filename
                                    crop_filename = f"{base_name}_frame{frame_idx}_det{detection_idx}_{label}_{conf:.2f}.jpg"
                                    crop_path = os.path.join(crops_dir, crop_filename)
                                    cv2.imwrite(crop_path, crop)
                                    detection_idx += 1
                            
                        min_detections_in_frame = min(min_detections_in_frame, num_detections_in_current_frame)
                        max_detections_in_frame = max(max_detections_in_frame, num_detections_in_current_frame)
                    
                    if not has_detections:
                        min_detections_in_frame = 0
                    
                    detection_value = 1 if has_detections else 0
                    summary_file.write(f"{file_name},{detection_value},{min_detections_in_frame},{max_detections_in_frame}\n")
                    
                    # Sort files into positive/negative detection folders if enabled
                    if sort_detections:
                        if has_detections:
                            shutil.copy2(file_path, os.path.join(positive_dir, file_name))
                        else:
                            shutil.copy2(file_path, os.path.join(negative_dir, file_name))
                    
                    if has_detections:
                        detections_count += 1
                    else:
                        no_detections_count += 1
            
            summary = f"Processing completed: {detections_count} files with detections, {no_detections_count} files without detections. Summary saved to {summary_file_path}"
            self.progress_updated.emit(len(files), len(files), summary)
            self.detection_completed.emit(summary)
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            error_msg = f"Error during processing: {str(e)}"
            self.error_occurred.emit(error_msg)

class DetectionApp(QWidget): # Renamed class
    progress_updated = pyqtSignal(int, int, str)
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.input_files = []
        self.input_type = 'none'
        # Determine initial detection type key for loading default output folder
        # This assumes 'body_detection' might be a primary default if nothing else is set
        # Or we can make it depend on the first item in model_type_combo
        initial_detection_type = self.config.get('general', 'last_used_model_type', 'body_detection') # Example
        self.current_detection_type_key = initial_detection_type # Set it early

        self.output_folder = self.config.get(
            'general', 
            'default_detection_output', 
            os.path.join(os.path.expanduser('~'), 'mfid', 'output', 'detections') # Fallback default
        )
        os.makedirs(self.output_folder, exist_ok=True) # Ensure default output folder exists
        self.current_file_index = 0
        self.total_files = 0
        self.progress_notification = None
        
        self.worker = DetectionWorker()
        self.worker_thread = None
        
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.detection_completed.connect(self.on_detection_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        self.setWindowTitle("Detection") # Changed window title
        self.resize(800, 750) # Adjusted size slightly
        self.init_ui()
        
        self.progress_updated.connect(self.update_progress)
        self.detect_best_device()

        # Populate models based on default model type
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed) # Connect first
        self.update_model_dropdown() # Then populate, which also loads settings

    def init_ui(self):
        DarkTheme(self) 
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(0)
        scroll_area.setStyleSheet("""
            QScrollArea { background-color: transparent; border: none; }
            QScrollBar:vertical { background-color: #333333; width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background-color: #555555; min-height: 20px; border-radius: 6px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        
        header_layout = QVBoxLayout()
        title_label = QLabel("Object Detection") # Changed title label
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        description_label = QLabel("Detect objects (e.g., bodies, faces) in videos/images using YOLO models") # Changed description
        description_label.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        description_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        main_layout.addLayout(header_layout)
        
        input_group = self.create_input_selection_group()
        main_layout.addWidget(input_group)
        
        self.files_info_label = QLabel("No files selected")
        self.files_info_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        self.files_info_label.setMaximumHeight(100)
        self.files_info_label.setWordWrap(True)
        
        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setFrameShape(0)
        files_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; max-height: 100px; }")
        files_scroll.setWidget(self.files_info_label)
        main_layout.addWidget(files_scroll)
        
        settings_group = self.create_settings_group()
        main_layout.addWidget(settings_group)
        
        progress_group = self.create_progress_group()
        main_layout.addWidget(progress_group)
        
        run_button_layout = QHBoxLayout()
        self.run_button = DarkButton('Run Detection', self.run_detection)
        self.run_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; border-radius: 4px; }
            QPushButton:hover { background-color: #43A047; }
            QPushButton:pressed { background-color: #388E3C; }
        """)
        run_button_layout.addStretch()
        run_button_layout.addWidget(self.run_button)
        run_button_layout.addStretch()
        main_layout.addLayout(run_button_layout)
        
        main_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll_area)
        
        # Connect signals for saving settings when UI elements change (optional, but good for persistence)
        # These connections have been moved to the end of init_ui to ensure all widgets are created.
        self.model_combo.currentTextChanged.connect(lambda: self.save_specific_setting('model', self.model_combo.currentText()))
        self.confidence_spinbox.valueChanged.connect(lambda val: self.save_specific_setting('confidence_threshold', val))
        self.iou_spinbox.valueChanged.connect(lambda val: self.save_specific_setting('iou_threshold', val))
        self.frame_skip_spinbox.valueChanged.connect(lambda val: self.save_specific_setting('frame_skip', val))
        self.save_videos_images_checkbox.stateChanged.connect(lambda state: self.save_specific_setting_checkbox('save_videos_images', bool(state), 'body_detection', 'save_videos', 'face_detection', 'save_full_frames'))
        self.save_txt_checkbox.stateChanged.connect(lambda state: self.save_specific_setting_checkbox('save_txt', bool(state), 'body_detection', 'save_txt', 'face_detection', 'save_coordinates'))
        self.device_combo.currentTextChanged.connect(lambda text: self.save_specific_setting('preferred_device', self.get_device_param()))
        # Imgsz combo connection could be added if it needs to be saved per detection type
        # self.imgsz_combo.currentTextChanged.connect(lambda text: self.save_specific_setting('imgsz', int(text)))

        logger.info("DetectionApp UI initialized.")

    def create_settings_group(self):
        settings_group = QGroupBox("Step 2: Configure Detection Settings")
        settings_group.setStyleSheet("""
            QGroupBox { border: 1px solid #555555; border-radius: 6px; margin-top: 12px; font-weight: bold; padding: 10px; background-color: #272727; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #2196F3; }
        """)
        form_layout = QFormLayout(settings_group)
        form_layout.setContentsMargins(15, 25, 15, 15) # Add some padding
        form_layout.setSpacing(10)

        # Apply scroll event filter to combo boxes and spin boxes
        scroll_filter = ScrollEventFilter(self)

        # Model Type Selection
        self.model_type_combo = QComboBox(self)
        self.model_type_combo.addItems(["Body Detection", "Face Detection"])
        self.model_type_combo.setToolTip("Select the type of model to use for detection.")
        self.model_type_combo.installEventFilter(scroll_filter)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed) # Connect to update models
        form_layout.addRow(QLabel("Model Type:"), self.model_type_combo)

        # Model Selection (now dynamically populated)
        self.model_combo = QComboBox(self)
        self.model_combo.setToolTip("Select the specific model file.")
        self.model_combo.installEventFilter(scroll_filter)
        form_layout.addRow(QLabel("Model:"), self.model_combo)

        # Device Selection (moved up for visibility)
        self.device_combo = QComboBox(self)
        self.device_label = QLabel("Device:")
        form_layout.addRow(self.device_label, self.device_combo)

        # Confidence Threshold
        self.confidence_spinbox = QDoubleSpinBox(self)
        self.confidence_spinbox.setRange(0.01, 1.00)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.25)
        self.confidence_spinbox.setToolTip("Minimum confidence score for a detection to be considered valid.")
        self.confidence_spinbox.installEventFilter(scroll_filter)
        form_layout.addRow(QLabel("Confidence Threshold:"), self.confidence_spinbox)

        # IOU Threshold
        self.iou_spinbox = QDoubleSpinBox(self)
        self.iou_spinbox.setRange(0.01, 1.00)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setToolTip("Intersection Over Union threshold for Non-Maximum Suppression.")
        self.iou_spinbox.installEventFilter(scroll_filter)
        form_layout.addRow(QLabel("IOU Threshold:"), self.iou_spinbox)
        
        # Image Size (optional, can be important for some models)
        self.imgsz_combo = QComboBox(self)
        self.imgsz_combo.addItems(["320", "416", "512", "640", "768", "1024", "1280"]) # Common sizes
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setToolTip("Input image size for the model (e.g., 640 for 640x640).")
        self.imgsz_combo.installEventFilter(scroll_filter)
        form_layout.addRow(QLabel("Image Size (imgsz):"), self.imgsz_combo)

        # Frame Skip (primarily for video)
        self.frame_skip_spinbox = QSpinBox(self)
        self.frame_skip_spinbox.setRange(0, 1000) # 0 means no skip
        self.frame_skip_spinbox.setValue(0)
        self.frame_skip_spinbox.setToolTip("Number of frames to skip between detections (for videos). 0 processes all frames.")
        self.frame_skip_spinbox.installEventFilter(scroll_filter)
        form_layout.addRow(QLabel("Frame Skip (videos):"), self.frame_skip_spinbox)
        
        # Output Options
        output_options_group = QGroupBox("Output Options")
        output_options_group.setFlat(True)
        output_options_layout = QVBoxLayout(output_options_group)

        # Save Summary - Only this one checked by default
        self.save_summary_checkbox = QCheckBox("Save Detection Summary", self)
        self.save_summary_checkbox.setChecked(True)
        self.save_summary_checkbox.setToolTip("Save a summary file with detection counts and statistics")
        output_options_layout.addWidget(self.save_summary_checkbox)

        # Sort into Positive/Negative folders
        self.sort_detections_checkbox = QCheckBox("Sort Files into Positive/Negative Detection Folders", self)
        self.sort_detections_checkbox.setChecked(False)
        self.sort_detections_checkbox.setToolTip("Copy input files into positive_detections and negative_detections folders based on detection results")
        output_options_layout.addWidget(self.sort_detections_checkbox)

        # Save Annotated Media
        self.save_videos_images_checkbox = QCheckBox("Save Annotated Media", self)
        self.save_videos_images_checkbox.setChecked(False)
        self.save_videos_images_checkbox.setToolTip("Save videos/images with detection boxes drawn")
        output_options_layout.addWidget(self.save_videos_images_checkbox)

        # Save Coordinates
        self.save_txt_checkbox = QCheckBox("Save Detection Coordinates", self)
        self.save_txt_checkbox.setChecked(False)
        self.save_txt_checkbox.setToolTip("Save detection coordinates to text files")
        output_options_layout.addWidget(self.save_txt_checkbox)

        # Save Cropped Detections
        self.save_crops_checkbox = QCheckBox("Save Cropped Detections", self)
        self.save_crops_checkbox.setChecked(False)
        self.save_crops_checkbox.setToolTip("Save cropped images of each detection")
        output_options_layout.addWidget(self.save_crops_checkbox)

        form_layout.addRow(output_options_group)

        return settings_group

    def update_model_dropdown(self):
        current_model_type = self.model_type_combo.currentText()
        
        self.model_combo.clear()
        
        # Define base paths for models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, '..', '..')) 
        
        if current_model_type == "Body Detection":
            model_base_path = os.path.join(project_root, 'mfid', 'models', 'body_detection')
            self.current_detection_type_key = 'body_detection'
        elif current_model_type == "Face Detection":
            model_base_path = os.path.join(project_root, 'mfid', 'models', 'face_detection')
            self.current_detection_type_key = 'face_detection'
        else:
            logger.warning(f"Unknown model type: {current_model_type}")
            self.current_detection_type_key = None
            return

        os.makedirs(model_base_path, exist_ok=True)

        try:
            # Get all .pt model files
            models = [f for f in os.listdir(model_base_path) if f.endswith('.pt')]
            
            # Sort models by size (nano, small, medium, large)
            def get_model_size(model_name):
                if 'nano' in model_name or '_n' in model_name:
                    return 0
                elif 'small' in model_name or '_s' in model_name:
                    return 1
                elif 'medium' in model_name or '_m' in model_name:
                    return 2
                elif 'large' in model_name or '_l' in model_name:
                    return 3
                return -1  # For unknown sizes
                
            models.sort(key=get_model_size)
            
            if models:
                self.model_combo.addItems(models)
                # Try to select the largest model by default
                for model in reversed(models):  # Start from end to find largest first
                    if any(size in model.lower() for size in ['large', '_l', 'medium', '_m', 'small', '_s', 'nano', '_n']):
                        self.model_combo.setCurrentText(model)
                        break
                else:  # If no size found in names, use last model
                    self.model_combo.setCurrentIndex(len(models) - 1)
            else:
                self.model_combo.addItem("No models found in directory")
                logger.warning(f"No .pt models found in {model_base_path}")
        except FileNotFoundError:
            self.model_combo.addItem("Model directory not found")
            logger.error(f"Model directory not found: {model_base_path}")
        except Exception as e:
            self.model_combo.addItem("Error loading models")
            logger.error(f"Error listing models in {model_base_path}: {e}")
        
        self.load_detection_settings()  # Load settings after models are populated

    def load_detection_settings(self):
        if not hasattr(self, 'current_detection_type_key') or not self.current_detection_type_key:
            logger.warning("Cannot load settings: current_detection_type_key not set.")
            return

        logger.info(f"Loading settings for {self.current_detection_type_key}")
        
        # Load common settings
        self.confidence_spinbox.setValue(self.config.get(self.current_detection_type_key, 'confidence_threshold', 0.25))
        self.iou_spinbox.setValue(self.config.get(self.current_detection_type_key, 'iou_threshold', 0.45))
        # Imgsz might be general or specific, here assuming general for simplicity or add to config
        # self.imgsz_combo.setCurrentText(str(self.config.get(self.current_detection_type_key, 'imgsz', 640)))

        # Always keep summary checked by default, others unchecked
        self.save_summary_checkbox.setChecked(True)
        self.save_videos_images_checkbox.setChecked(False)
        self.save_txt_checkbox.setChecked(False)
        self.save_crops_checkbox.setChecked(False)
        self.sort_detections_checkbox.setChecked(False)

        if self.current_detection_type_key == 'body_detection':
            self.frame_skip_spinbox.setValue(self.config.get('body_detection', 'frame_skip', 0))
            self.frame_skip_spinbox.setEnabled(True)
        elif self.current_detection_type_key == 'face_detection':
            self.frame_skip_spinbox.setValue(self.config.get('face_detection', 'frame_skip', 5))
            self.frame_skip_spinbox.setEnabled(True)
        
        # Preferred device might be global or per type. Assuming per type for now.
        preferred_device = self.config.get(self.current_detection_type_key, 'preferred_device', 'cpu')
        if self.device_combo.findText(preferred_device) != -1:
            self.device_combo.setCurrentText(preferred_device)

        # Model - already handled in update_model_dropdown to some extent
        saved_model = self.config.get(self.current_detection_type_key, 'model')
        if saved_model and self.model_combo.findText(saved_model) != -1:
            self.model_combo.setCurrentText(saved_model)
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)

    def save_detection_settings(self):
        if not hasattr(self, 'current_detection_type_key') or not self.current_detection_type_key:
            logger.warning("Cannot save settings: current_detection_type_key not set.")
            return

        logger.info(f"Saving settings for {self.current_detection_type_key}")

        self.config.set(self.current_detection_type_key, 'model', self.model_combo.currentText())
        self.config.set(self.current_detection_type_key, 'confidence_threshold', self.confidence_spinbox.value())
        self.config.set(self.current_detection_type_key, 'iou_threshold', self.iou_spinbox.value())
        # self.config.set(self.current_detection_type_key, 'imgsz', int(self.imgsz_combo.currentText())) # If storing imgsz per type
        self.config.set(self.current_detection_type_key, 'preferred_device', self.get_device_param())

        if self.current_detection_type_key == 'body_detection':
            self.config.set('body_detection', 'save_videos', self.save_videos_images_checkbox.isChecked())
            self.config.set('body_detection', 'save_txt', self.save_txt_checkbox.isChecked())
            self.config.set('body_detection', 'frame_skip', self.frame_skip_spinbox.value())
        elif self.current_detection_type_key == 'face_detection':
            self.config.set('face_detection', 'save_full_frames', self.save_videos_images_checkbox.isChecked()) # Map from UI
            self.config.set('face_detection', 'save_coordinates', self.save_txt_checkbox.isChecked()) # Map from UI
            self.config.set('face_detection', 'frame_skip', self.frame_skip_spinbox.value())
        
        # self.config.save_config() # .set() already saves if ConfigManager is implemented that way

    def create_input_selection_group(self):
        input_group = QGroupBox("Step 1: Select Input & Output")
        input_group.setStyleSheet("""
            QGroupBox { border: 1px solid #555555; border-radius: 6px; margin-top: 12px; font-weight: bold; padding: 10px; background-color: #272727; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #2196F3; }
        """)
        input_layout = QFormLayout(input_group)
        input_layout.setContentsMargins(15, 25, 15, 15)
        input_layout.setSpacing(10)

        # Input source selection
        input_source_layout = QHBoxLayout()
        self.files_radio = QRadioButton("Select File(s)")
        self.folder_radio = QRadioButton("Select Folder")
        self.files_radio.setChecked(True) # Default to file selection
        self.input_type_group = QButtonGroup(self)
        self.input_type_group.addButton(self.files_radio)
        self.input_type_group.addButton(self.folder_radio)
        input_source_layout.addWidget(self.files_radio)
        input_source_layout.addWidget(self.folder_radio)
        input_source_layout.addStretch()
        input_layout.addRow(QLabel("Input Source:"), input_source_layout)

        # Browse button
        self.browse_button = DarkButton("Browse...", self.browse_files_or_folder)
        input_layout.addRow(self.browse_button)
        
        # Clear selection button
        self.clear_button = DarkButton("Clear Selection", self.clear_selection)
        input_layout.addRow(self.clear_button)

        # Output folder selection
        output_folder_layout = QVBoxLayout()
        
        # Main output folder
        main_output_layout = QHBoxLayout()
        self.output_folder_label = QLabel(f"Output to: {self.output_folder}")
        self.output_folder_label.setWordWrap(True)
        self.select_output_button = DarkButton("Change Output Folder", self.select_output_folder)
        main_output_layout.addWidget(self.output_folder_label)
        main_output_layout.addWidget(self.select_output_button)
        output_folder_layout.addLayout(main_output_layout)
        
        # Custom subfolder name
        subfolder_layout = QHBoxLayout()
        self.custom_folder_checkbox = QCheckBox("Custom Detection Folder Name")
        self.custom_folder_checkbox.setChecked(False)
        self.custom_folder_name = QLineEdit()
        self.custom_folder_name.setPlaceholderText("Enter custom folder name (optional)")
        self.custom_folder_name.setEnabled(False)
        self.custom_folder_checkbox.stateChanged.connect(lambda state: self.custom_folder_name.setEnabled(state == Qt.Checked))
        subfolder_layout.addWidget(self.custom_folder_checkbox)
        subfolder_layout.addWidget(self.custom_folder_name)
        output_folder_layout.addLayout(subfolder_layout)
        
        input_layout.addRow(output_folder_layout)

        return input_group

    def create_progress_group(self):
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet("""
            QGroupBox { border: 1px solid #444444; border-radius: 4px; margin-top: 8px; font-weight: bold; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #cccccc; padding: 5px;")
        progress_layout.addWidget(self.status_label)
        return progress_group

    def detect_best_device(self):
        self.device_combo.clear()
        try:
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                detected_device = "mps"
                self.device_label.setText("Device (Apple MPS detected):")
                logger.info("Apple MPS (Metal) detected")
            elif torch.cuda.is_available():
                detected_device = f"cuda:{torch.cuda.current_device()}"
                self.device_label.setText(f"Device (CUDA GPU detected: {torch.cuda.get_device_name(torch.cuda.current_device())}):")
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                detected_device = "cpu"
                self.device_label.setText("Device (CPU only):")
                logger.info("No GPU detected, using CPU.")
            
            self.device_combo.addItems([detected_device, "cpu"]) # Offer detected and CPU
            if detected_device != "cpu" and detected_device != "mps": # if CUDA
                 # Allow selection of specific CUDA devices if multiple are present
                for i in range(torch.cuda.device_count()):
                    if f"cuda:{i}" not in [detected_device, "cpu"]: # Avoid duplicates
                        self.device_combo.addItem(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
            
            self.device_combo.setCurrentText(detected_device)
        except Exception as e:
            logger.error(f"Error detecting device: {e}", exc_info=True)
            self.device_label.setText("Device (Error detecting):")
            self.device_combo.addItems(["cpu"])
            self.device_combo.setCurrentText("cpu")

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video/Image Files", "", 
                                                "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All Files (*)")
        if files:
            self.input_files = files
            self.input_type = 'file'
            self.update_files_info()
            logger.info(f"{len(files)} file(s) selected.")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Videos/Images")
        if folder:
            self.input_files = self.find_supported_files(folder)
            self.input_type = 'folder'
            self.update_files_info()
            logger.info(f"Folder selected: {folder}. Found {len(self.input_files)} supported files.")
    
    def find_supported_files(self, folder):
        supported_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png')
        return [os.path.join(folder, f) for f in os.listdir(folder) 
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(supported_extensions)]

    def update_files_info(self):
        if not self.input_files:
            self.files_info_label.setText("No files selected")
            return
        
        if self.input_type == 'file':
            if len(self.input_files) == 1:
                self.files_info_label.setText(f"Selected file: {os.path.basename(self.input_files[0])}")
            else:
                self.files_info_label.setText(f"Selected {len(self.input_files)} files. Hover to see all.")
                self.files_info_label.setToolTip("\n".join([os.path.basename(f) for f in self.input_files]))
        elif self.input_type == 'folder':
            folder_path = os.path.dirname(self.input_files[0]) if self.input_files else "Unknown"
            self.files_info_label.setText(f"Selected folder: {folder_path} ({len(self.input_files)} files). Hover to see all.")
            self.files_info_label.setToolTip("\n".join([os.path.basename(f) for f in self.input_files]))
        self.total_files = len(self.input_files)
        self.progress_bar.setMaximum(self.total_files if self.total_files > 0 else 100)

    def browse_files_or_folder(self):
        if self.files_radio.isChecked():
            self.select_files()
        elif self.folder_radio.isChecked():
            self.select_folder()

    def run_detection(self):
        if not self.input_files:
            Notifications.warning(self, "No input files selected. Please select files or a folder.")
            return
        if not self.output_folder:
            Notifications.warning(self, "No output folder selected. Please select an output folder.")
            return

        selected_model_file = self.model_combo.currentText()
        if not selected_model_file or "No models found" in selected_model_file or "Error loading" in selected_model_file or "Model directory not found" in selected_model_file:
            Notifications.warning(self, "No valid model selected. Please check model configuration.")
            return

        # Check if a detection is already running
        if self.worker_thread and self.worker_thread.is_alive():
            Notifications.warning(self, "A detection process is already running. Please wait for it to complete.")
            return

        # Save current settings before running
        self.save_detection_settings()

        # Create timestamped or custom output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        selected_type = self.model_type_combo.currentText()
        detection_type = "face" if selected_type == "Face Detection" else "body"
        
        if self.custom_folder_checkbox.isChecked() and self.custom_folder_name.text().strip():
            custom_name = self.custom_folder_name.text().strip()
            detection_folder_name = f"{custom_name}_{timestamp}"
        else:
            detection_folder_name = f"{detection_type}_detections_{timestamp}"
        
        # Create output folder directly in the output directory
        detection_output_path = os.path.join(self.output_folder, detection_folder_name)
        os.makedirs(detection_output_path, exist_ok=True)

        # Determine model base path from model type
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
        if selected_type == "Body Detection":
            model_base_path = os.path.join(project_root, 'mfid', 'models', 'body_detection')
        elif selected_type == "Face Detection":
            model_base_path = os.path.join(project_root, 'mfid', 'models', 'face_detection')
        else:
            Notifications.error(self, f"Invalid model type selected: {selected_type}")
            return

        params = {
            'conf': self.confidence_spinbox.value(),
            'iou': self.iou_spinbox.value(),
            'device': self.get_device_param(),
            'imgsz': int(self.imgsz_combo.currentText()),
            'save_summary': self.save_summary_checkbox.isChecked(),
            'save_videos_images': self.save_videos_images_checkbox.isChecked(),
            'save_txt': self.save_txt_checkbox.isChecked(),
            'save_crops': self.save_crops_checkbox.isChecked(),
            'sort_detections': self.sort_detections_checkbox.isChecked(),
            'frame_skip': self.frame_skip_spinbox.value(),
        }
        
        logger.info(f"Starting detection with parameters: {params}")
        self.status_label.setText("Status: Processing...")
        self.progress_bar.setValue(0)
        
        self.worker_thread = threading.Thread(target=self.worker.process_files, 
                                            args=(self.input_files, params, detection_output_path, model_base_path, selected_model_file), 
                                            daemon=True)
        self.worker_thread.start()

        # Use the correct show_progress class method
        self.progress_notification = ProgressNotification.show_progress(self, "Processing files...")

    def get_device_param(self):
        device_text = self.device_combo.currentText()
        if "mps" in device_text:
            return "mps"
        elif "cuda" in device_text:
            # Extract "cuda:0", "cuda:1", etc.
            return device_text.split(' ')[0] 
        return "cpu"

    @pyqtSlot(str)
    def on_detection_completed(self, summary):
        self.status_label.setText(f"Status: {summary}")
        Notifications.success(self, f"Detection Completed: {summary}")
        logger.info(f"Detection completed: {summary}")
        if self.progress_notification:
            self.progress_notification.close()

    @pyqtSlot(str)
    def on_error_occurred(self, error_msg):
        self.status_label.setText(f"Status: Error - {error_msg}")
        Notifications.error(self, f"Detection Error: {error_msg}")
        logger.error(f"Detection error reported to UI: {error_msg}")
        if self.progress_notification:
            self.progress_notification.close()
        # Optionally re-enable run button or offer retry
        # self.run_button.setEnabled(True) 

    @pyqtSlot(int, int, str)
    def update_progress(self, current, total, message):
        self.current_file_index = current
        self.total_files = total
        
        if total > 0:
            progress_percent = int((current / total) * 100)
            self.progress_bar.setValue(progress_percent)
        else:
            self.progress_bar.setValue(0)  # Or some indeterminate state if preferred

        if current == total and "completed" in message.lower():  # Final message
             self.status_label.setText(f"Status: {message}")
        else:
            self.status_label.setText(f"Status: Processing {current}/{total} - {message}")

        if self.progress_notification:
            # Update progress notification with percentage and message
            self.progress_notification.update_progress(progress_percent, message)

        QApplication.processEvents()  # Keep UI responsive

    def clear_selection(self):
        self.input_files = []
        self.input_type = 'none'
        self.update_files_info()
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Ready")
        logger.info("Input selection cleared.")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder)
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(f"Output to: {self.output_folder}")
            # Save this as the new default using the correct config key and set method
            self.config.set('general', 'default_detection_output', folder)
            # self.config.save_config() # .set() already saves
            logger.info(f"Output folder changed to: {folder}")

    def on_model_type_changed(self, model_type_text):
        self.update_model_dropdown() # This will repopulate models and then load settings for the new type

    def save_specific_setting(self, key, value):
        if hasattr(self, 'current_detection_type_key') and self.current_detection_type_key:
            self.config.set(self.current_detection_type_key, key, value)
            # logger.debug(f"Saved {key}={value} for {self.current_detection_type_key}")

    def save_specific_setting_checkbox(self, ui_element_meaning, value, type1_key, config_key1, type2_key, config_key2):
        """ Handles saving checkbox state to the correct config key based on model type. """
        if hasattr(self, 'current_detection_type_key') and self.current_detection_type_key:
            if self.current_detection_type_key == type1_key:
                self.config.set(type1_key, config_key1, value)
                # logger.debug(f"Saved {config_key1}={value} for {type1_key}")
            elif self.current_detection_type_key == type2_key:
                self.config.set(type2_key, config_key2, value)
                # logger.debug(f"Saved {config_key2}={value} for {type2_key}")

if __name__ == '__main__':
    # Ensure QApplication instance exists for event posting
    app = QApplication.instance() 
    if app is None: 
        app = QApplication(sys.argv)
    
    # Create dummy model directories if they don't exist for testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_models = os.path.normpath(os.path.join(script_dir, '..', '..'))
    
    dummy_body_models_path = os.path.join(project_root_for_models, 'mfid', 'models', 'body_detection')
    dummy_face_models_path = os.path.join(project_root_for_models, 'mfid', 'models', 'face_detection')
    os.makedirs(dummy_body_models_path, exist_ok=True)
    os.makedirs(dummy_face_models_path, exist_ok=True)
    
    # Create dummy model files for testing dropdown
    if not os.path.exists(os.path.join(dummy_body_models_path, 'yolov8n_body_test.pt')):
        with open(os.path.join(dummy_body_models_path, 'yolov8n_body_test.pt'), 'w') as f:
            f.write("dummy model") # Dummy content
    if not os.path.exists(os.path.join(dummy_face_models_path, 'yolov8n_face_test.pt')):
        with open(os.path.join(dummy_face_models_path, 'yolov8n_face_test.pt'), 'w') as f:
            f.write("dummy model") # Dummy content

    window = DetectionApp()
    window.show()
    sys.exit(app.exec_()) 