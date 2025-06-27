import os
import threading
import cv2
import glob
import shutil
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, 
                          QMessageBox, QLabel, QComboBox, QLineEdit, QProgressBar, QGroupBox,
                          QScrollArea, QHBoxLayout, QRadioButton, QButtonGroup, QFormLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit
from mfid.utils.notifications import Notifications, ProgressNotification
from mfid.utils.config_manager import ConfigManager
from mfid.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger('identity_app')

class IdentityApp(QWidget):
    # Add progress signal for updates
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.input_files = []
        self.input_type = 'none'  # 'file', 'folder', or 'none'
        self.output_folder = ''
        self.current_file_index = 0
        self.total_files = 0
        self.progress_notification = None
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Monkey Identity Detection')
        self.resize(800, 600)
        
        # Create a scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(0)  # No frame
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #333333;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Create a container widget for the scrollable content
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        
        # Header with title and description
        header_layout = QVBoxLayout()
        title_label = QLabel("Monkey Identity Detection")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        description_label = QLabel("Detect and identify individual monkeys in videos and images")
        description_label.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        description_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        main_layout.addLayout(header_layout)
        
        # Input selection - Step 1
        input_group = self.create_input_selection_group()
        main_layout.addWidget(input_group)
        
        # Selected files info with fixed height
        self.files_info_label = QLabel("No files selected")
        self.files_info_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        self.files_info_label.setMaximumHeight(100)  # Set maximum height to prevent excessive expansion
        self.files_info_label.setWordWrap(True)  # Allow word wrapping
        
        # Make files info scrollable if it gets too long
        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setFrameShape(0)  # No frame
        files_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
                max-height: 100px;
            }
        """)
        files_scroll.setWidget(self.files_info_label)
        main_layout.addWidget(files_scroll)
        
        # Settings - Step 2
        settings_group = self.create_settings_group()
        settings_group.setTitle("Step 2: Configure Detection Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                padding: 10px;
                background-color: #272727;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2196F3;
            }
        """)
        main_layout.addWidget(settings_group)
        
        # Progress information
        progress_group = self.create_progress_group()
        main_layout.addWidget(progress_group)
        
        # Run button - Step 3
        run_container = QGroupBox("Step 3: Run Identity Detection")
        run_container.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                padding: 10px;
                background-color: #272727;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2196F3;
            }
        """)
        run_layout = QVBoxLayout()
        
        instructions_label = QLabel("Click the button below to start identity detection")
        instructions_label.setStyleSheet("font-style: italic; color: #aaaaaa; margin-bottom: 5px;")
        run_layout.addWidget(instructions_label)
        
        self.run_button = DarkButton("Run Identity Detection", self.runDetection)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        run_layout.addWidget(self.run_button)
        run_container.setLayout(run_layout)
        main_layout.addWidget(run_container)
        
        # Set the scroll area's widget to our container
        scroll_area.setWidget(scroll_content)
        
        # Create the main window layout and add the scroll area
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll_area)
        
        # Set a reasonable fixed size for the window
        self.setFixedSize(800, 700)
        
        # Connect signals
        self.progress_updated.connect(self.update_progress)

    def create_input_selection_group(self):
        """Create the input selection group box"""
        input_group = QGroupBox("Step 1: Select Input Files/Folder")
        input_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                padding: 10px;
                background-color: #272727;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2196F3;
            }
        """)
        
        input_layout = QVBoxLayout()
        
        # Radio buttons
        radio_layout = QHBoxLayout()
        self.radio_group = QButtonGroup(self)
        
        self.radio_file = QRadioButton("Individual Files")
        self.radio_folder = QRadioButton("Folder")
        self.radio_group.addButton(self.radio_file)
        self.radio_group.addButton(self.radio_folder)
        
        # Set the first option as default
        self.radio_file.setChecked(True)
        
        radio_layout.addWidget(self.radio_file)
        radio_layout.addWidget(self.radio_folder)
        radio_layout.addStretch()
        
        input_layout.addLayout(radio_layout)
        
        # Browse button - make it prominent
        browse_button = DarkButton("Browse for Files/Folder", self.browse_files_or_folder)
        browse_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                font-size: 14px;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """)
        
        input_layout.addWidget(browse_button)
        
        # Clear Selection button
        clear_button = DarkButton("Clear Selection", self.clear_selection)
        clear_button.setStyleSheet("""
            QPushButton {
                padding: 6px;
                background-color: #555555;
                color: white;
                margin-bottom: 5px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        input_layout.addWidget(clear_button)
        
        input_group.setLayout(input_layout)
        
        return input_group
    
    def create_settings_group(self):
        """Create the settings group box"""
        settings_group = QGroupBox("Detection Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        settings_layout = QVBoxLayout()
        
        # Add info about available models instead of dropdowns
        models_info = QLabel("Using available models: Face Detection (Large) and Identity (Medium)")
        models_info.setStyleSheet("font-weight: normal; font-style: italic; color: #aaaaaa; margin-bottom: 10px;")
        settings_layout.addWidget(models_info)
        
        # Output folder selection
        output_folder_layout = QHBoxLayout()
        output_folder_label = QLabel("Output Folder:")
        output_folder_label.setStyleSheet("font-weight: normal;")
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setReadOnly(True)  # Display only
        self.output_folder_edit.setStyleSheet("""
            QLineEdit { 
                background-color: #E0E0E0; 
                color: black; 
                border: 1px solid #CCCCCC; 
                padding: 6px; 
                border-radius: 3px; 
                font-size: 13px;
            } 
        """)
        # Update placeholder to reflect the default location
        default_output_path_display = ".../mfid/output/identity_detections"
        self.output_folder_edit.setPlaceholderText(f"Default: {default_output_path_display}")
        browse_output_button = DarkButton("Browse", self.select_output_folder)
        
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(browse_output_button)
        settings_layout.addLayout(output_folder_layout)
        
        # Add custom folder name option
        custom_name_layout = QHBoxLayout()
        custom_name_label = QLabel("Custom Folder Name:")
        custom_name_label.setStyleSheet("font-weight: normal;")
        self.custom_name_edit = QLineEdit()
        self.custom_name_edit.setPlaceholderText("Optional - Leave empty for timestamp only")
        self.custom_name_edit.setStyleSheet("""
            QLineEdit { 
                background-color: #E0E0E0; 
                color: black; 
                border: 1px solid #CCCCCC; 
                padding: 6px; 
                border-radius: 3px; 
                font-size: 13px;
            } 
            QLineEdit::placeholder { 
                color: #666666; 
            }
        """)
        
        custom_name_layout.addWidget(custom_name_label)
        custom_name_layout.addWidget(self.custom_name_edit)
        settings_layout.addLayout(custom_name_layout)
        
        # Add frame skipping option
        frame_skip_layout = QHBoxLayout()
        frame_skip_label = QLabel("Frame Skip Rate:")
        frame_skip_label.setStyleSheet("font-weight: normal;")
        self.frame_skip_edit = QLineEdit()
        self.frame_skip_edit.setPlaceholderText("1 (default, no skip)")
        self.frame_skip_edit.setStyleSheet("""
            QLineEdit { 
                background-color: #E0E0E0; 
                color: black; 
                border: 1px solid #CCCCCC; 
                padding: 6px; 
                border-radius: 3px; 
                font-size: 13px;
            } 
            QLineEdit::placeholder { 
                color: #666666; 
            }
        """)
        self.frame_skip_edit.setText("1")  # Default value
        frame_skip_info = QLabel("(1 = every frame, 2 = every 2nd frame, etc.)")
        frame_skip_info.setStyleSheet("font-style: italic; color: #aaaaaa;")
        
        frame_skip_layout.addWidget(frame_skip_label)
        frame_skip_layout.addWidget(self.frame_skip_edit)
        frame_skip_layout.addWidget(frame_skip_info)
        settings_layout.addLayout(frame_skip_layout)
        
        # Additional options
        save_layout = QVBoxLayout()
        save_label = QLabel("Output Options:")
        save_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        save_layout.addWidget(save_label)
        
        self.save_annotated_box = QCheckBox("Save annotated videos/images with bounding boxes")
        self.save_annotated_box.setChecked(True)
        save_layout.addWidget(self.save_annotated_box)
        
        self.save_summary_box = QCheckBox("Generate summary file with identity scores")
        self.save_summary_box.setChecked(True)
        save_layout.addWidget(self.save_summary_box)
        
        self.delete_faces_box = QCheckBox("Delete extracted faces after processing")
        self.delete_faces_box.setChecked(True)
        save_layout.addWidget(self.delete_faces_box)
        
        settings_layout.addLayout(save_layout)
        settings_group.setLayout(settings_layout)
        
        return settings_group
    
    def create_progress_group(self):
        """Create the progress information group box"""
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #E0E0E0;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
                margin: 0.5px;
                border-radius: 3px;
            }
        """)
        
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        
        return progress_group
    
    def select_output_folder(self):
        """Open a dialog to select the output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_edit.setText(folder)
            logger.info(f"Output folder selected: {folder}")
        else:
            # If user cancels, clear the selection and revert to default indication
            self.output_folder = ''
            self.output_folder_edit.setText("")  # Clear display to show placeholder again
            logger.info("Output folder selection cancelled, using default.")
    
    def browse_files_or_folder(self):
        """Browse for files or folder based on the selected radio button"""
        if self.radio_file.isChecked():
            self.select_files()
        else:
            self.select_folder()
    
    def select_files(self):
        """Select individual files for processing"""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Files", 
            "", 
            "Video/Image Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png)",
            options=options
        )
        
        if files:
            self.input_files = files
            self.input_type = 'file'
            self.update_files_info()
            
            # Save to recent files in config if needed
            for file in files:
                self.config.add_recent_file(file)
    
    def select_folder(self):
        """Select a folder containing videos/images"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Videos/Images")
        
        if folder:
            self.input_type = 'folder'
            self.input_folder = folder
            
            # Get all supported files in the folder
            self.input_files = self.find_supported_files(folder)
            self.update_files_info()
            
            # Save to recent directories in config
            self.config.add_recent_directory(folder)
    
    def find_supported_files(self, folder):
        """Find all supported video and image files in the folder"""
        supported_extensions = (".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png")
        files = []
        
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith(supported_extensions):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def update_files_info(self):
        """Update the files information label"""
        self.total_files = len(self.input_files)
        
        if self.total_files == 0:
            self.files_info_label.setText("No files selected")
            return
        
        # Group by type
        videos = [f for f in self.input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
        images = [f for f in self.input_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        info_text = f"Selected {self.total_files} files: {len(videos)} videos, {len(images)} images"
        
        # Limit the number of files displayed to avoid excessive UI height
        if self.total_files <= 3:
            file_list = ", ".join([os.path.basename(f) for f in self.input_files])
            info_text += f"\nFiles: {file_list}"
        else:
            sample_files = ", ".join([os.path.basename(f) for f in self.input_files[:3]])
            info_text += f"\nSample files: {sample_files} and {self.total_files - 3} more"
        
        self.files_info_label.setText(info_text)
    
    def clear_selection(self):
        """Clear the current file selection"""
        self.input_files = []
        self.update_files_info()
        Notifications.info(self, "File selection cleared")
    
    def runDetection(self):
        """Prepare and start identity detection"""
        # Validate input
        if not self.input_files:
            Notifications.warning(self, "Please select files or folder first")
            return
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get custom name if provided
        custom_name = self.custom_name_edit.text().strip()
        folder_name = f"identity_detections_{timestamp}"
        if custom_name:
            folder_name = f"{custom_name}_{timestamp}"
        
        # Determine output folder: Use selected or default to project's output directory
        if self.output_folder:
            # If custom output folder is selected, use it directly
            output_dir = os.path.join(self.output_folder, folder_name)
        else:
            # Default to output folder in the workspace root
            workspace_root = "/Users/quentinbacquele/mfid"
            output_dir = os.path.join(workspace_root, "output", folder_name)  # Directly in output folder
            logger.info(f"No output folder selected, defaulting to: {output_dir}")
        
        # Check if the determined output directory exists, create if not
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating output directory {output_dir}: {e}")
            Notifications.error(self, f"Could not create output directory: {output_dir}\nError: {e}")
            return
        
        # Update the output folder display
        self.output_folder_edit.setText(output_dir)
        
        # Assign the determined and verified path to self.output_folder for use later
        self.output_folder = output_dir
        
        # Reset progress
        self.current_file_index = 0
        self.total_files = len(self.input_files)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting identity detection...")
        
        # Show progress notification
        self.progress_notification = ProgressNotification.show_progress(
            self, f"Processing {self.total_files} files..."
        )
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(
            target=self.processFiles,
            args=(self.input_files,),
            daemon=True
        ).start()
    
    def processFiles(self, file_paths):
        """Process all files for identity detection"""
        try:
            for i, file_path in enumerate(file_paths):
                # Update progress
                self.current_file_index = i + 1
                self.progress_updated.emit(self.current_file_index, self.total_files, os.path.basename(file_path))
                
                # Process each file
                cumulative_scores = {}
                success = self.processDetection(file_path, cumulative_scores)
                if success and cumulative_scores:
                    self.saveCumulativeResults(cumulative_scores, file_path)
            
            # Complete progress
            self.progress_updated.emit(self.total_files, self.total_files, "Processing completed!")
            
        except Exception as e:
            logger.error(f"Error in processFiles: {e}", exc_info=True)
            self.progress_updated.emit(-1, -1, f"Error: {str(e)}")
    
    def processDetection(self, file_path, cumulative_scores):
        """Process a single file for identity detection"""
        try:
            logger.info(f"Processing file: {file_path}")
            # Notify UI about current file
            QApplication.processEvents()
            
            # Get frame skip rate
            try:
                frame_skip = max(1, int(self.frame_skip_edit.text()))
            except ValueError:
                frame_skip = 1
                logger.warning("Invalid frame skip value, using default (1)")
            
            # Determine if file is an image or a video
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
            video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI']
            file_extension = os.path.splitext(file_path)[1].lower()
            
            is_video = file_extension in video_extensions
            is_image = file_extension in image_extensions
            
            # Get model paths
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'models'))
            face_models_dir = os.path.join(models_dir, 'face_detection')
            identity_models_dir = os.path.join(models_dir)
            
            # Hardcoded model sizes since only these are available
            face_model_size = 'l'
            identity_model_size = 'm'
            
            path_to_face_model = os.path.join(face_models_dir, f'best_{face_model_size}_face.pt')
            path_to_identity_model = os.path.join(identity_models_dir, f'best_{identity_model_size}_id.pt')
            
            # Check if models exist
            if not os.path.exists(path_to_face_model):
                logger.error(f"Face model not found: {path_to_face_model}")
                self.progress_updated.emit(-1, -1, f"Face model not found: {path_to_face_model}")
                return False
                
            if not os.path.exists(path_to_identity_model):
                logger.error(f"Identity model not found: {path_to_identity_model}")
                self.progress_updated.emit(-1, -1, f"Identity model not found: {path_to_identity_model}")
                return False
            
            # Load models
            face_model = YOLO(path_to_face_model)
            identity_model = YOLO(path_to_identity_model)
            
            # Create a unique subfolder for this file's output
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_folder = os.path.join(self.output_folder, base_name)
            os.makedirs(save_folder, exist_ok=True)
            
            # Output file paths
            txt_file_path = os.path.join(save_folder, f"{base_name}_identity.txt")
            with open(txt_file_path, "w") as txt_file:
                txt_file.write("Frame\tConfs\tNames\n")  # Updated header to include frame number
                
                if is_video:
                    output_file_name = f"{base_name}_identity.mp4"
                    output_file_path = os.path.join(save_folder, output_file_name)
                    
                    cap = cv2.VideoCapture(file_path)
                    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_width = int(original_width * 0.5)
                    frame_height = int(original_height * 0.5)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'X264')
                    out = cv2.VideoWriter(output_file_path, fourcc, frame_rate, (frame_width, frame_height))
                    
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_count += 1
                        # Skip frames based on frame_skip value
                        if frame_count % frame_skip != 0:
                            continue
                            
                        # Update progress more frequently for videos
                        if frame_count % 30 == 0:  # Update every 30 frames
                            progress = (frame_count * 100) // total_frames
                            self.progress_updated.emit(progress, 100, f"Processing frame {frame_count}/{total_frames}")
                        
                        # Resize frame for faster processing
                        frame = cv2.resize(frame, (frame_width, frame_height))
                        
                        # Process frame with face detection
                        face_result = face_model(frame)[0]
                        
                        # Process each face in the frame
                        for box in face_result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cropped_img = frame[y1:y2, x1:x2]
                            cropped_img_path = os.path.join(save_folder, f'face_{frame_count}_{x1}_{y1}.jpg')
                            cv2.imwrite(cropped_img_path, cropped_img)
                            
                            # Identify the face
                            identity_results = identity_model(cropped_img_path)
                            for identity_result in identity_results:
                                for idx, conf in zip(identity_result.probs.top5, identity_result.probs.top5conf):
                                    label = identity_result.names[idx]
                                    if label not in cumulative_scores:
                                        cumulative_scores[label] = []
                                    cumulative_scores[label].append(conf)
                                    
                                idx1 = identity_result.probs.top1
                                idx5 = identity_result.probs.top5
                                conf1 = identity_result.probs.top1conf.item()
                                conf5 = identity_result.probs.top5conf.tolist()
                                names_dict = identity_result.names
                                conf_name = names_dict[idx1]
                                conf_names = [names_dict[i] for i in idx5]
                                
                                txt_file.write(f"{frame_count}\t{conf5}\t{conf_names}\n")
                                
                                # Draw rectangle and text on the frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                text = f'{conf_name}: {conf1:.2f}'
                                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            if self.delete_faces_box.isChecked():
                                os.remove(cropped_img_path)
                        
                        out.write(frame)
                    
                    cap.release()
                    out.release()
                
                else:  # Process single image
                    frame = cv2.imread(file_path)
                    
                    # Process frame with face detection
                    face_result = face_model(frame)[0]
                    
                    # Process each face in the frame
                    for box in face_result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cropped_img = frame[y1:y2, x1:x2]
                        cropped_img_path = os.path.join(save_folder, f'face_{x1}_{y1}.jpg')
                        cv2.imwrite(cropped_img_path, cropped_img)
                        
                        # Identify the face
                        identity_results = identity_model(cropped_img_path)
                        for identity_result in identity_results:
                            for idx, conf in zip(identity_result.probs.top5, identity_result.probs.top5conf):
                                label = identity_result.names[idx]
                                if label not in cumulative_scores:
                                    cumulative_scores[label] = []
                                cumulative_scores[label].append(conf)
                                
                            idx1 = identity_result.probs.top1
                            idx5 = identity_result.probs.top5
                            conf1 = identity_result.probs.top1conf.item()
                            conf5 = identity_result.probs.top5conf.tolist()
                            names_dict = identity_result.names
                            conf_name = names_dict[idx1]
                            conf_names = [names_dict[i] for i in idx5]
                            
                            txt_file.write(f"{conf5}\t{conf_names}\n")
                            
                            # Draw rectangle and text on the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            text = f'{conf_name}: {conf1:.2f}'
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        if self.delete_faces_box.isChecked():
                            os.remove(cropped_img_path)
            
            # Delete cropped face images if option is selected
            if self.delete_faces_box.isChecked():
                files_to_remove = glob.glob(os.path.join(save_folder, 'face_*.jpg'))
                for file in files_to_remove:
                    os.remove(file)
            
            logger.info(f"Completed processing: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            self.progress_updated.emit(-1, -1, f"Error processing {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def saveCumulativeResults(self, cumulative_scores, file_path):
        """Save summary results with top identities and confidence scores"""
        if not self.save_summary_box.isChecked() or not cumulative_scores:
            return
            
        try:
            # Calculate the mean accuracy for each label
            mean_scores = {label: sum(scores) / len(scores) for label, scores in cumulative_scores.items()}
            # Sort the results by highest mean score
            sorted_mean_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_folder = os.path.join(self.output_folder, base_name)
            os.makedirs(save_folder, exist_ok=True)
            summary_file_path = os.path.join(save_folder, f"{base_name}_summary.txt")
            
            with open(summary_file_path, "w") as f:
                f.write("Label\tMean Accuracy\n")
                for label, mean_score in sorted_mean_scores:
                    f.write(f"{label}\t{mean_score:.4f}\n")
                    
            logger.info(f"Saved summary for {file_path} to {summary_file_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary for {file_path}: {e}", exc_info=True)
    
    @pyqtSlot(int, int, str)
    def update_progress(self, current, total, message):
        """Update progress bar and status label"""
        if current == -1 and total == -1:
            # This is an error message
            self.status_label.setText(f"Error: {message}")
            Notifications.error(self, message)
            return
            
        if total > 0:
            percentage = (current * 100) // total
            self.progress_bar.setValue(percentage)
            
        self.status_label.setText(message)

# Example usage (for testing standalone)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = IdentityApp()
    window.show()
    sys.exit(app.exec_())
