import os
import threading
import cv2
import glob
import shutil
import torch
import datetime
import multiprocessing
import platform
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, 
                          QMessageBox, QLabel, QComboBox, QLineEdit, QProgressBar, QGroupBox,
                          QScrollArea, QHBoxLayout, QRadioButton, QButtonGroup, QFormLayout, 
                          QCheckBox, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QEvent
from PyQt5.QtGui import QFont
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit
from mfid.utils.notifications import Notifications, ProgressNotification
from mfid.utils.config_manager import ConfigManager
from mfid.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger('analysis_app')

# Event filter to ignore scroll wheel events on specific widgets
class ScrollEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)

# Worker class to handle processing in a separate thread
class AnalysisWorker(QObject):
    progress_updated = pyqtSignal(int, int, str)
    analysis_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def process_files(self, files, params, output_path, analysis_mode, models_info):
        try:
            if analysis_mode == 'classification':
                self._process_classification(files, params, output_path, models_info)
            elif analysis_mode == 'detection':
                self._process_detection(files, params, output_path, models_info)
            elif analysis_mode == 'detection_classification':
                self._process_detection_classification(files, params, output_path, models_info)
        except Exception as e:
            logger.error(f"Error in analysis processing: {e}")
            self.error_occurred.emit(str(e))
            
    def _process_classification(self, files, params, output_path, models_info):
        """Process files for direct classification"""
        model_path = models_info['classification_model']
        logger.info(f"Loading classification model: {model_path}")
        
        if not os.path.exists(model_path):
            self.error_occurred.emit(f"Classification model not found: {model_path}")
            return
            
        model = YOLO(model_path)
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        results_dir = os.path.join(output_path, "classification_results")
        os.makedirs(results_dir, exist_ok=True)
        
        summary_file_path = os.path.join(output_path, "classification_summary.txt")
        with open(summary_file_path, 'w') as summary_file:
            summary_file.write("filename,predicted_class,confidence\n")
            
            for i, file_path in enumerate(files):
                file_name = os.path.basename(file_path)
                self.progress_updated.emit(i+1, len(files), f"Classifying {file_name}")
                
                # Classify the image
                results = model.predict(file_path, conf=params.get('conf', 0.5))
                result = results[0]
                
                if result.probs is not None:
                    top_class_id = result.probs.top1
                    confidence = result.probs.top1conf.item()
                    class_name = result.names[top_class_id]
                    
                    summary_file.write(f"{file_name},{class_name},{confidence:.4f}\n")
                    logger.info(f"Classified {file_name} as {class_name} with confidence {confidence:.4f}")
                else:
                    summary_file.write(f"{file_name},unknown,0.0000\n")
                    
        self.analysis_completed.emit(f"Classification completed. Results saved to {output_path}")
        
    def _process_detection(self, files, params, output_path, models_info):
        """Process files for direct detection (full implementation from original detection app)"""
        model_path = models_info['detection_model']
        logger.info(f"Loading detection model: {model_path}")
        
        if not os.path.exists(model_path):
            self.error_occurred.emit(f"Detection model not found: {model_path}")
            return
            
        model = YOLO(model_path)
        
        save_summary = params.get('save_summary', True)
        save_videos_images = params.get('save_videos_images', False)
        save_txt = params.get('save_txt', False)
        save_crops = params.get('save_crops', False)
        sort_detections = params.get('sort_detections', False)
        frame_skip = params.get('frame_skip', 0)

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
                    'conf': params.get('conf', 0.5),
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

                        # Save crops if enabled
                        if save_crops and len(result.boxes) > 0:
                            detection_idx = 0
                            for box in result.boxes:
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

                    # Save crops for images
                    if save_crops and len(result.boxes) > 0:
                        frame = result.orig_img
                        detection_idx = 0
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = result.names[cls]
                            
                            # Extract the detection crop
                            crop = frame[y1:y2, x1:x2].copy()
                            
                            # Save with informative filename
                            crop_filename = f"{base_name}_det{detection_idx}_{label}_{conf:.2f}.jpg"
                            crop_path = os.path.join(crops_dir, crop_filename)
                            cv2.imwrite(crop_path, crop)
                            detection_idx += 1
                
                has_detections = False
                min_detections_in_frame = float('inf')
                max_detections_in_frame = 0
                
                for result in results:
                    boxes = result.boxes
                    num_detections_in_current_frame = len(boxes)
                    
                    if num_detections_in_current_frame > 0:
                        has_detections = True
                        
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
        self.analysis_completed.emit(summary)
        
    def _process_detection_classification(self, files, params, output_path, models_info):
        """Process files for two-step detection then classification (full implementation like identity app)"""
        face_model_path = models_info['detection_model']
        identity_model_path = models_info['classification_model']
        
        logger.info(f"Loading face detection model: {face_model_path}")
        logger.info(f"Loading identity classification model: {identity_model_path}")
        
        if not os.path.exists(face_model_path):
            self.error_occurred.emit(f"Face detection model not found: {face_model_path}")
            return
        if not os.path.exists(identity_model_path):
            self.error_occurred.emit(f"Identity model not found: {identity_model_path}")
            return
            
        face_model = YOLO(face_model_path)
        identity_model = YOLO(identity_model_path)
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        
        frame_skip = max(1, params.get('frame_skip', 1))  # Ensure at least 1 to prevent division by zero
        save_annotated = params.get('save_annotated', False)
        save_summary = params.get('save_summary', True)
        delete_faces = params.get('delete_faces', True)
        
        cumulative_scores = {}
        
        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            self.progress_updated.emit(i+1, len(files), f"Processing {file_name}")
            
            # Create subfolder for this file
            save_folder = os.path.join(output_path, base_name)
            os.makedirs(save_folder, exist_ok=True)
            
            try:
                # Determine if file is an image or a video
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
                video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI']
                file_extension = os.path.splitext(file_path)[1].lower()
                
                is_video = file_extension in video_extensions
                is_image = file_extension in image_extensions
                
                # Output file paths
                txt_file_path = os.path.join(save_folder, f"{base_name}_identity.txt")
                
                with open(txt_file_path, "w") as txt_file:
                    if is_video:
                        txt_file.write("Frame\tConfs\tNames\n")
                    else:
                        txt_file.write("Confs\tNames\n")
                        
                    if is_video:
                        # Process video
                        if save_annotated:
                            output_file_name = f"{base_name}_identity.mp4"
                            output_file_path = os.path.join(save_folder, output_file_name)
                        
                        cap = cv2.VideoCapture(file_path)
                        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame_width = int(original_width * 0.5)
                        frame_height = int(original_height * 0.5)
                        
                        out = None
                        if save_annotated:
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
                                self.progress_updated.emit(i+1, len(files), f"Processing frame {frame_count}/{total_frames}")
                            
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
                                    # Get all probabilities and sort them to get top 20 (or all available)
                                    all_probs = identity_result.probs.data.cpu().numpy()
                                    names_dict = identity_result.names
                                    
                                    # Create list of (index, confidence) pairs and sort by confidence
                                    prob_pairs = [(i, float(conf)) for i, conf in enumerate(all_probs)]
                                    prob_pairs.sort(key=lambda x: x[1], reverse=True)
                                    
                                    # Get top 20 (or all available if less than 20)
                                    top_n = min(20, len(prob_pairs))
                                    top_indices = [pair[0] for pair in prob_pairs[:top_n]]
                                    top_confs = [pair[1] for pair in prob_pairs[:top_n]]
                                    top_names = [names_dict[i] for i in top_indices]
                                    
                                    # Store all top predictions in cumulative scores
                                    for idx, conf in zip(top_indices, top_confs):
                                        label = names_dict[idx]
                                        if label not in cumulative_scores:
                                            cumulative_scores[label] = []
                                        cumulative_scores[label].append(conf)
                                    
                                    # For display, get top1 
                                    idx1 = identity_result.probs.top1
                                    conf1 = identity_result.probs.top1conf.item()
                                    conf_name = names_dict[idx1]
                                    
                                    # Write top 20 to file instead of just top 5
                                    txt_file.write(f"{frame_count}\t{top_confs}\t{top_names}\n")
                                    
                                    # Draw rectangle and text on the frame if saving annotated
                                    if save_annotated:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        text = f'{conf_name}: {conf1:.2f}'
                                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                
                                if delete_faces:
                                    os.remove(cropped_img_path)
                            
                            if save_annotated and out:
                                out.write(frame)
                        
                        cap.release()
                        if out:
                            out.release()
                    
                    else:  # Process single image
                        frame = cv2.imread(file_path)
                        if frame is None:
                            continue
                            
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
                                # Get all probabilities and sort them to get top 20 (or all available)
                                all_probs = identity_result.probs.data.cpu().numpy()
                                names_dict = identity_result.names
                                
                                # Create list of (index, confidence) pairs and sort by confidence
                                prob_pairs = [(i, float(conf)) for i, conf in enumerate(all_probs)]
                                prob_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                # Get top 20 (or all available if less than 20)
                                top_n = min(20, len(prob_pairs))
                                top_indices = [pair[0] for pair in prob_pairs[:top_n]]
                                top_confs = [pair[1] for pair in prob_pairs[:top_n]]
                                top_names = [names_dict[i] for i in top_indices]
                                
                                # Store all top predictions in cumulative scores
                                for idx, conf in zip(top_indices, top_confs):
                                    label = names_dict[idx]
                                    if label not in cumulative_scores:
                                        cumulative_scores[label] = []
                                    cumulative_scores[label].append(conf)
                                
                                # For backward compatibility, still get top1 for display
                                idx1 = identity_result.probs.top1
                                conf1 = identity_result.probs.top1conf.item()
                                conf_name = names_dict[idx1]
                                
                                # Write top 20 to file instead of just top 5
                                txt_file.write(f"{top_confs}\t{top_names}\n")
                                
                                # Draw rectangle and text on the frame if saving annotated
                                if save_annotated:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    text = f'{conf_name}: {conf1:.2f}'
                                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            if delete_faces:
                                os.remove(cropped_img_path)
                        
                        # Save annotated image if requested
                        if save_annotated:
                            annotated_img_path = os.path.join(save_folder, f"{base_name}_identity.jpg")
                            cv2.imwrite(annotated_img_path, frame)
                                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Save cumulative results
        if save_summary:
            self._save_cumulative_results(cumulative_scores, output_path)
        
        self.analysis_completed.emit(f"Identity detection completed. Results saved to {output_path}")
        
    def _save_cumulative_results(self, cumulative_scores, output_path):
        """Save cumulative identity results"""
        if not cumulative_scores:
            return
            
        try:
            # Calculate mean accuracy for each label
            mean_scores = {label: sum(scores) / len(scores) for label, scores in cumulative_scores.items()}
            # Show top 20 results (or max available if less than 20)
            max_results = min(20, len(mean_scores))
            sorted_mean_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
            
            summary_file_path = os.path.join(output_path, "identity_summary.txt")
            with open(summary_file_path, "w") as f:
                f.write("Label\tMean Accuracy\n")
                for label, mean_score in sorted_mean_scores:
                    f.write(f"{label}\t{mean_score:.4f}\n")
                    
        except Exception as e:
            logger.error(f"Error saving cumulative results: {e}")


class AnalysisApp(QWidget):
    progress_updated = pyqtSignal(int, int, str)
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.input_files = []
        self.input_type = 'none'
        # Set default output folder to mfid directory
        self.output_folder = os.path.join(os.path.dirname(__file__), '..', '..')
        self.output_folder = os.path.abspath(self.output_folder)  # Get absolute path
        self.analysis_mode = 'detection'  # 'classification', 'detection', 'detection_classification'
        self.worker = None
        self.thread = None
        self.progress_notification = None
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('MFID Analysis - Detection, Classification & Identity Recognition')
        self.resize(800, 700)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(0)
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
        
        # Create container for scrollable content
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        
        # Header
        header_layout = QVBoxLayout()
        title_label = QLabel("MFID Analysis")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        description_label = QLabel("Comprehensive image and video analysis: detection, classification, and identity recognition")
        description_label.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        description_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        main_layout.addLayout(header_layout)
        
        # Step 1: Input/Output/Custom Name
        step1_group = self.create_input_output_group()
        main_layout.addWidget(step1_group)
        
        # Selected files info
        self.files_info_label = QLabel("No files selected")
        self.files_info_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        self.files_info_label.setMaximumHeight(100)
        self.files_info_label.setWordWrap(True)
        main_layout.addWidget(self.files_info_label)
        
        # Step 2: Analysis Mode Selection
        step2_group = self.create_analysis_mode_group()
        main_layout.addWidget(step2_group)
        
        # Step 3: Mode-specific Options
        self.step3_group = self.create_options_group()
        main_layout.addWidget(self.step3_group)
        
        # Progress
        progress_group = self.create_progress_group()
        main_layout.addWidget(progress_group)
        
        # Run button
        run_group = self.create_run_group()
        main_layout.addWidget(run_group)
        
        # Set up scroll area
        scroll_area.setWidget(scroll_content)
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll_area)
        
        # Connect signals
        self.progress_updated.connect(self.update_progress)
        
        # Now it's safe to update the options UI
        self.update_options_ui()
        
        # Load initial settings
        self.load_settings()

    def create_input_output_group(self):
        """Create Step 1: Input/Output/Custom Name selection"""
        group = QGroupBox("Step 1: Input, Output & Project Settings")
        group.setStyleSheet("""
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
        layout = QVBoxLayout(group)
        
        # Input selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        
        self.input_button_group = QButtonGroup()
        self.file_radio = QRadioButton("Single File")
        self.folder_radio = QRadioButton("Folder")
        self.file_radio.setChecked(True)
        
        self.input_button_group.addButton(self.file_radio)
        self.input_button_group.addButton(self.folder_radio)
        
        input_layout.addWidget(self.file_radio)
        input_layout.addWidget(self.folder_radio)
        
        self.browse_input_button = DarkButton("Browse...", self.browse_input)
        input_layout.addWidget(self.browse_input_button)
        input_layout.addStretch()
        
        layout.addLayout(input_layout)
        
        # Output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_button = DarkButton("Browse...", self.select_output_folder)
        output_layout.addWidget(self.output_folder_button)
        output_layout.addStretch()
        layout.addLayout(output_layout)
        
        self.output_folder_label = QLabel(self.output_folder)
        self.output_folder_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        self.output_folder_label.setWordWrap(True)
        layout.addWidget(self.output_folder_label)
        
        # Project name
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project Name:"))
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("e.g., monkey_analysis_2024")
        DarkLineEdit(self.project_name_input, "e.g., monkey_analysis_2024")
        project_layout.addWidget(self.project_name_input)
        layout.addLayout(project_layout)
        
        return group

    def create_analysis_mode_group(self):
        """Create Step 2: Analysis Mode Selection"""
        group = QGroupBox("Step 2: Analysis Mode")
        group.setStyleSheet("""
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
        layout = QVBoxLayout(group)
        
        # Analysis mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Detection - Find objects in images/videos",
            "Classification - Classify entire images", 
            "Detection + Classification - Detect faces then identify individuals"
        ])
        self.mode_combo.currentTextChanged.connect(self.on_analysis_mode_changed)
        layout.addWidget(self.mode_combo)
        
        # Mode descriptions
        self.mode_description = QLabel()
        self.mode_description.setStyleSheet("color: #aaaaaa; font-style: italic; margin-top: 10px;")
        self.mode_description.setWordWrap(True)
        layout.addWidget(self.mode_description)
        
        # Initialize mode description
        self.analysis_mode = 'detection'
        self.mode_description.setText("Detect and locate objects (faces, bodies, etc.) in images and videos. Outputs bounding box coordinates and detection confidence scores.")
        
        return group

    def create_options_group(self):
        """Create Step 3: Analysis Configuration (static with enable/disable)"""
        group = QGroupBox("Step 3: Analysis Configuration")
        group.setStyleSheet("""
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
        layout = QVBoxLayout(group)
        
        # === SHARED OPTIONS (Available for all analysis modes) ===
        shared_group = QGroupBox("Shared Options")
        shared_layout = QFormLayout(shared_group)
        
        # Create scroll event filter
        scroll_filter = ScrollEventFilter(self)
        
        # Confidence Threshold (shared by all modes)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.5)
        self.conf_spinbox.setToolTip("Minimum confidence score for predictions")
        self.conf_spinbox.installEventFilter(scroll_filter)
        shared_layout.addRow("Confidence Threshold:", self.conf_spinbox)
        
        # Device Selection (shared by all modes)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setCurrentText("auto")
        self.device_combo.setToolTip("Processing device (auto = best available)")
        self.device_combo.installEventFilter(scroll_filter)
        shared_layout.addRow("Device:", self.device_combo)
        
        # Image Size (shared by all modes)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "768", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setToolTip("Input image size for models")
        self.imgsz_combo.installEventFilter(scroll_filter)
        shared_layout.addRow("Image Size:", self.imgsz_combo)
        
        layout.addWidget(shared_group)
        
        # Add spacing
        spacer1 = QLabel()
        spacer1.setMaximumHeight(10)
        layout.addWidget(spacer1)
        
        # === DETECTION OPTIONS (from Detection App) ===
        detection_group = QGroupBox("Detection Options")
        detection_layout = QFormLayout(detection_group)
        
        # Model Type Selection (from detection app)
        model_type_layout = QHBoxLayout()
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Body Detection", "Face Detection"])
        self.model_type_combo.setToolTip("Select the type of model to use for detection")
        self.model_type_combo.currentTextChanged.connect(self.update_detection_models)
        self.model_type_combo.installEventFilter(scroll_filter)
        model_type_layout.addWidget(self.model_type_combo)
        detection_layout.addRow("Model Type:", model_type_layout)
        
        # Detection Model
        detection_model_layout = QHBoxLayout()
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.installEventFilter(scroll_filter)
        self.update_detection_models()
        detection_model_layout.addWidget(self.detection_model_combo)
        
        # Add import button for detection models
        self.import_detection_model_btn = DarkButton("+", self.import_detection_model)
        self.import_detection_model_btn.setFixedSize(30, 30)
        self.import_detection_model_btn.setToolTip("Import custom detection model (.pt file)")
        detection_model_layout.addWidget(self.import_detection_model_btn)
        
        detection_layout.addRow("Detection Model:", detection_model_layout)
        
        # IOU Threshold (detection only)
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.01, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setToolTip("Intersection Over Union threshold for Non-Maximum Suppression")
        self.iou_spinbox.installEventFilter(scroll_filter)
        detection_layout.addRow("IOU Threshold:", self.iou_spinbox)
        
        # Frame Skip (for videos)
        self.frame_skip_spinbox = QSpinBox()
        self.frame_skip_spinbox.setRange(0, 1000)
        self.frame_skip_spinbox.setValue(0)
        self.frame_skip_spinbox.setToolTip("Number of frames to skip between detections (0 = process all frames)")
        self.frame_skip_spinbox.installEventFilter(scroll_filter)
        detection_layout.addRow("Frame Skip (videos):", self.frame_skip_spinbox)
        
        layout.addWidget(detection_group)
        
        # Add spacing
        spacer2 = QLabel()
        spacer2.setMaximumHeight(10)
        layout.addWidget(spacer2)
        
        # === CLASSIFICATION OPTIONS ===
        classification_group = QGroupBox("Classification Options")
        classification_layout = QFormLayout(classification_group)
        
        # Classification Model
        classification_model_layout = QHBoxLayout()
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.installEventFilter(scroll_filter)
        self.update_classification_models()
        classification_model_layout.addWidget(self.classification_model_combo)
        
        # Add import button for classification models
        self.import_classification_model_btn = DarkButton("+", self.import_classification_model)
        self.import_classification_model_btn.setFixedSize(30, 30)
        self.import_classification_model_btn.setToolTip("Import custom classification model (.pt file)")
        classification_model_layout.addWidget(self.import_classification_model_btn)
        
        classification_layout.addRow("Classification Model:", classification_model_layout)
        
        layout.addWidget(classification_group)
        
        # Add spacing
        spacer3 = QLabel()
        spacer3.setMaximumHeight(10)
        layout.addWidget(spacer3)
        
        # === DETECTION OUTPUT OPTIONS (from Detection App) ===
        detection_output_group = QGroupBox("Detection Output Options")
        detection_output_layout = QVBoxLayout(detection_output_group)
        
        # From Detection App
        self.save_detection_summary_cb = QCheckBox("Save Detection Summary")
        self.save_detection_summary_cb.setChecked(True)
        self.save_detection_summary_cb.setToolTip("Save a summary file with detection counts and statistics")
        detection_output_layout.addWidget(self.save_detection_summary_cb)
        
        self.sort_detections_cb = QCheckBox("Sort Files into Positive/Negative Detection Folders")
        self.sort_detections_cb.setChecked(False)
        self.sort_detections_cb.setToolTip("Copy input files into positive_detections and negative_detections folders based on detection results")
        detection_output_layout.addWidget(self.sort_detections_cb)
        
        self.save_annotated_media_cb = QCheckBox("Save Annotated Media")
        self.save_annotated_media_cb.setChecked(False)
        self.save_annotated_media_cb.setToolTip("Save videos/images with detection boxes drawn")
        detection_output_layout.addWidget(self.save_annotated_media_cb)
        
        self.save_detection_coords_cb = QCheckBox("Save Detection Coordinates")
        self.save_detection_coords_cb.setChecked(False)
        self.save_detection_coords_cb.setToolTip("Save detection coordinates to text files")
        detection_output_layout.addWidget(self.save_detection_coords_cb)
        
        self.save_crops_cb = QCheckBox("Save Cropped Detections")
        self.save_crops_cb.setChecked(False)
        self.save_crops_cb.setToolTip("Save cropped images of each detection")
        detection_output_layout.addWidget(self.save_crops_cb)
        
        layout.addWidget(detection_output_group)
        
        # Add spacing
        spacer4 = QLabel()
        spacer4.setMaximumHeight(10)
        layout.addWidget(spacer4)
        
        # === CLASSIFICATION OUTPUT OPTIONS ===
        classification_output_group = QGroupBox("Classification Output Options")
        classification_output_layout = QVBoxLayout(classification_output_group)
        
        self.save_classification_summary_cb = QCheckBox("Generate classification summary file")
        self.save_classification_summary_cb.setChecked(True)
        classification_output_layout.addWidget(self.save_classification_summary_cb)
        
        layout.addWidget(classification_output_group)
        
        # Add spacing
        spacer5 = QLabel()
        spacer5.setMaximumHeight(10)
        layout.addWidget(spacer5)
        
        # === DETECTION + CLASSIFICATION OUTPUT OPTIONS (from Identity App) ===
        identity_output_group = QGroupBox("Detection + Classification Output Options")
        identity_output_layout = QVBoxLayout(identity_output_group)
        
        # From Identity App
        self.save_annotated_identity_cb = QCheckBox("Save annotated videos/images with bounding boxes and classifications")
        self.save_annotated_identity_cb.setChecked(True)
        identity_output_layout.addWidget(self.save_annotated_identity_cb)
        
        self.save_identity_summary_cb = QCheckBox("Generate summary file with classification results")
        self.save_identity_summary_cb.setChecked(True)
        identity_output_layout.addWidget(self.save_identity_summary_cb)
        
        self.delete_faces_cb = QCheckBox("Delete extracted detections after processing")
        self.delete_faces_cb.setChecked(True)
        identity_output_layout.addWidget(self.delete_faces_cb)
        
        layout.addWidget(identity_output_group)
        
        # Apply custom styling to make disabled state more obvious
        self.apply_custom_checkbox_styling()
        
        return group

    def apply_custom_checkbox_styling(self):
        """Apply custom styling to make enabled/disabled state more obvious"""
        checkbox_style = """
            QCheckBox {
                color: #ffffff;
                font-weight: normal;
                padding: 2px;
            }
            QCheckBox:enabled {
                color: #ffffff;
                background-color: transparent;
            }
            QCheckBox:disabled {
                color: #666666;
                background-color: #1a1a1a;
                border-radius: 3px;
                padding: 2px;
                font-style: italic;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #555555;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:enabled {
                border: 2px solid #2196F3;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:disabled {
                border: 2px solid #333333;
                background-color: #1a1a1a;
            }
            QCheckBox::indicator:checked:enabled {
                background-color: #2196F3;
                border: 2px solid #2196F3;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMi41TDQgNi41TDIgNC41IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #444444;
                border: 2px solid #333333;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMi41TDQgNi41TDIgNC41IiBzdHJva2U9IiM2NjY2NjYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }
        """
        
        # Apply styling to all checkboxes
        for checkbox in [
            # Detection output checkboxes
            self.save_detection_summary_cb,
            self.sort_detections_cb,
            self.save_annotated_media_cb,
            self.save_detection_coords_cb,
            self.save_crops_cb,
            # Classification output checkboxes
            self.save_classification_summary_cb,
            # Identity output checkboxes
            self.save_annotated_identity_cb,
            self.save_identity_summary_cb,
            self.delete_faces_cb
        ]:
            checkbox.setStyleSheet(checkbox_style)
        
        # Also apply to combo boxes and spin boxes for consistency
        combo_style = """
            QComboBox:enabled {
                color: #ffffff;
                background-color: #2a2a2a;
                border: 2px solid #2196F3;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox:disabled {
                color: #666666;
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 4px;
                padding: 4px;
            }
        """
        
        spinbox_style = """
            QDoubleSpinBox:enabled, QSpinBox:enabled {
                color: #ffffff;
                background-color: #2a2a2a;
                border: 2px solid #2196F3;
                border-radius: 4px;
                padding: 4px;
            }
            QDoubleSpinBox:disabled, QSpinBox:disabled {
                color: #666666;
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 4px;
                padding: 4px;
            }
        """
        
        # Apply to all combo boxes
        for combo in [self.model_type_combo, self.detection_model_combo, self.classification_model_combo, 
                     self.device_combo, self.imgsz_combo]:
            combo.setStyleSheet(combo_style)
            
        # Apply to all spin boxes
        for spinbox in [self.conf_spinbox, self.iou_spinbox, self.frame_skip_spinbox]:
            spinbox.setStyleSheet(spinbox_style)

    def create_progress_group(self):
        """Create progress display group"""
        group = QGroupBox("Progress")
        group.setStyleSheet("""
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
        layout = QVBoxLayout(group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to start analysis")
        self.progress_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.progress_label)
        
        return group

    def create_run_group(self):
        """Create run button group"""
        group = QGroupBox("Step 4: Run Analysis")
        group.setStyleSheet("""
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
        layout = QVBoxLayout(group)
        
        self.run_button = DarkButton("Start Analysis", self.run_analysis)
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
        layout.addWidget(self.run_button)
        
        return group

    def on_analysis_mode_changed(self, mode_text):
        """Handle analysis mode change"""
        if "Detection + Classification" in mode_text:
            self.analysis_mode = 'detection_classification'
            description = "Two-step process: First detect faces/objects, then classify each detection to identify individuals. Ideal for identity recognition tasks."
        elif "Detection -" in mode_text:
            self.analysis_mode = 'detection'
            description = "Detect and locate objects (faces, bodies, etc.) in images and videos. Outputs bounding box coordinates and detection confidence scores."
        elif "Classification -" in mode_text:
            self.analysis_mode = 'classification'
            description = "Classify entire images into predefined categories. Outputs class predictions and confidence scores for each image."
        else:
            # Fallback
            self.analysis_mode = 'detection'
            description = "Detect and locate objects (faces, bodies, etc.) in images and videos. Outputs bounding box coordinates and detection confidence scores."
            
        self.mode_description.setText(description)
        
        # Update classification models based on new mode
        self.update_classification_models()
        
        # Update UI options
        self.update_options_ui()

    def update_options_ui(self):
        """Update the options UI based on selected analysis mode"""
        # Shared options are always enabled
        # (conf_spinbox, device_combo, imgsz_combo are always available)
        
        if self.analysis_mode == 'detection':
            # Detection mode: Enable detection-specific options
            self.model_type_combo.setEnabled(True)
            self.detection_model_combo.setEnabled(True)
            self.classification_model_combo.setEnabled(False)
            self.iou_spinbox.setEnabled(True)
            self.frame_skip_spinbox.setEnabled(True)
            
            # Detection output options - all enabled
            self.save_detection_summary_cb.setEnabled(True)
            self.sort_detections_cb.setEnabled(True)
            self.save_annotated_media_cb.setEnabled(True)
            self.save_detection_coords_cb.setEnabled(True)
            self.save_crops_cb.setEnabled(True)
            
            # Classification output options - disabled
            self.save_classification_summary_cb.setEnabled(False)
            
            # Identity output options - disabled
            self.save_annotated_identity_cb.setEnabled(False)
            self.save_identity_summary_cb.setEnabled(False)
            self.delete_faces_cb.setEnabled(False)
            
            # Set tooltips for disabled options
            self.classification_model_combo.setToolTip("Not used in detection mode")
            self.save_classification_summary_cb.setToolTip("Only available in classification mode")
            self.save_annotated_identity_cb.setToolTip("Only available in detection+classification mode")
            self.save_identity_summary_cb.setToolTip("Only available in detection+classification mode")
            self.delete_faces_cb.setToolTip("Only available in detection+classification mode")
            
        elif self.analysis_mode == 'classification':
            # Classification mode: Enable classification-specific options
            self.model_type_combo.setEnabled(False)
            self.detection_model_combo.setEnabled(False)
            self.classification_model_combo.setEnabled(True)
            self.iou_spinbox.setEnabled(False)
            self.frame_skip_spinbox.setEnabled(False)
            
            # Detection output options - disabled
            self.save_detection_summary_cb.setEnabled(False)
            self.sort_detections_cb.setEnabled(False)
            self.save_annotated_media_cb.setEnabled(False)
            self.save_detection_coords_cb.setEnabled(False)
            self.save_crops_cb.setEnabled(False)
            
            # Classification output options - enabled
            self.save_classification_summary_cb.setEnabled(True)
            
            # Identity output options - disabled
            self.save_annotated_identity_cb.setEnabled(False)
            self.save_identity_summary_cb.setEnabled(False)
            self.delete_faces_cb.setEnabled(False)
            
            # Set tooltips for disabled options
            self.model_type_combo.setToolTip("Model type selection not needed for classification")
            self.detection_model_combo.setToolTip("Not used in classification mode")
            self.iou_spinbox.setToolTip("Only needed for object detection")
            self.frame_skip_spinbox.setToolTip("Frame skipping not relevant for classification")
            self.save_detection_summary_cb.setToolTip("No detections in classification mode")
            self.sort_detections_cb.setToolTip("No detections to sort in classification mode")
            self.save_annotated_media_cb.setToolTip("No annotations to save in classification mode")
            self.save_detection_coords_cb.setToolTip("No bounding boxes in classification mode")
            self.save_crops_cb.setToolTip("No objects to crop in classification mode")
            self.save_annotated_identity_cb.setToolTip("Only available in detection+classification mode")
            self.save_identity_summary_cb.setToolTip("Only available in detection+classification mode")
            self.delete_faces_cb.setToolTip("Only available in detection+classification mode")
            
        elif self.analysis_mode == 'detection_classification':
            # Two-step mode: Enable both detection and classification options (identity workflow)
            # Model selection - force face detection for identity workflow (like identity app)
            self.model_type_combo.setCurrentText("Face Detection")
            self.model_type_combo.setEnabled(False)  # Lock to face detection for identity
            self.update_detection_models()  # Update to show face detection models
            self.detection_model_combo.setEnabled(True)  # Face detection model
            self.classification_model_combo.setEnabled(True)  # Identity classification model
            self.iou_spinbox.setEnabled(True)
            self.frame_skip_spinbox.setEnabled(True)
            
            # Detection output options - all disabled (identity has its own workflow)
            self.save_detection_summary_cb.setEnabled(False)
            self.sort_detections_cb.setEnabled(False)
            self.save_annotated_media_cb.setEnabled(False)
            self.save_detection_coords_cb.setEnabled(False)
            self.save_crops_cb.setEnabled(False)
            
            # Classification output options - disabled (identity has its own)
            self.save_classification_summary_cb.setEnabled(False)
            
            # Identity output options - all enabled (this is the main workflow)
            self.save_annotated_identity_cb.setEnabled(True)
            self.save_identity_summary_cb.setEnabled(True)
            self.delete_faces_cb.setEnabled(True)
            
            # Set tooltips for disabled options
            self.save_detection_summary_cb.setToolTip("Use 'Generate summary file with identity scores' instead")
            self.sort_detections_cb.setToolTip("File sorting not applicable to identity recognition workflow")
            self.save_annotated_media_cb.setToolTip("Use 'Save annotated videos/images with bounding boxes' in Identity options instead")
            self.save_detection_coords_cb.setToolTip("Coordinates are included in identity results automatically")
            self.save_crops_cb.setToolTip("Faces are extracted and processed automatically - use 'Delete faces after processing' to control cleanup")
            self.save_classification_summary_cb.setToolTip("Use 'Generate summary file with identity scores' in Identity options instead")
            
        # Clear tooltips for enabled options
        all_widgets = [
            self.model_type_combo, self.detection_model_combo, self.classification_model_combo, 
            self.iou_spinbox, self.frame_skip_spinbox,
            self.save_detection_summary_cb, self.sort_detections_cb, self.save_annotated_media_cb,
            self.save_detection_coords_cb, self.save_crops_cb,
            self.save_classification_summary_cb,
            self.save_annotated_identity_cb, self.save_identity_summary_cb, self.delete_faces_cb
        ]
        for widget in all_widgets:
            if widget.isEnabled():
                widget.setToolTip("")

    def update_detection_models(self):
        """Update detection model dropdown based on selected model type"""
        if not hasattr(self, 'detection_model_combo') or not hasattr(self, 'model_type_combo'):
            return
            
        model_type = self.model_type_combo.currentText()
        
        # Determine subdirectory based on model type
        if model_type == "Body Detection":
            subdir = "body_detection"
        elif model_type == "Face Detection":
            subdir = "face_detection"
        else:
            subdir = "body_detection"  # Default
        
        # Get models from specific subdirectory
        models_base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        models_dir = os.path.join(models_base_dir, subdir)
        models = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pt'):
                    models.append(file)
        
        # Sort models by size (nano, small, medium, large)
        def get_model_size(model_name):
            if 'nano' in model_name.lower() or '_n' in model_name.lower():
                return 0
            elif 'small' in model_name.lower() or '_s' in model_name.lower():
                return 1
            elif 'medium' in model_name.lower() or '_m' in model_name.lower():
                return 2
            elif 'large' in model_name.lower() or '_l' in model_name.lower():
                return 3
            return -1
            
        models.sort(key=get_model_size)
        
        self.detection_model_combo.clear()
        if models:
            self.detection_model_combo.addItems(models)
            # Try to select the largest model by default
            for model in reversed(models):
                if any(size in model.lower() for size in ['large', '_l', 'medium', '_m']):
                    self.detection_model_combo.setCurrentText(model)
                    break
        else:
            self.detection_model_combo.addItem("No models found in directory")

    def update_classification_models(self):
        """Update classification model dropdown"""
        if not hasattr(self, 'classification_model_combo'):
            return
            
        models_base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        models = []
        
        if self.analysis_mode == 'detection_classification':
            # For identity mode, look for identity models (typically named with 'id' in them)
            if os.path.exists(models_base_dir):
                for file in os.listdir(models_base_dir):
                    if file.endswith('.pt') and ('id' in file.lower() or 'identity' in file.lower()):
                        models.append(file)
            
            # Sort by size if available
            def get_model_size(model_name):
                if 'nano' in model_name.lower() or '_n' in model_name.lower():
                    return 0
                elif 'small' in model_name.lower() or '_s' in model_name.lower():
                    return 1
                elif 'medium' in model_name.lower() or '_m' in model_name.lower():
                    return 2
                elif 'large' in model_name.lower() or '_l' in model_name.lower():
                    return 3
                return -1
                
            models.sort(key=get_model_size)
        else:
            # For regular classification, look for classification models
            if os.path.exists(models_base_dir):
                for file in os.listdir(models_base_dir):
                    if file.endswith('.pt') and '-cls' in file:
                        models.append(file)
            
            # Add default YOLOv8 classification models if none found
            if not models:
                models = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"]
        
        self.classification_model_combo.clear()
        if models:
            self.classification_model_combo.addItems(models)
            # Try to select medium model by default for identity
            if self.analysis_mode == 'detection_classification':
                for model in models:
                    if 'medium' in model.lower() or '_m' in model.lower():
                        self.classification_model_combo.setCurrentText(model)
                        break
        else:
            self.classification_model_combo.addItem("No models found")

    def browse_input(self):
        """Browse for input files or folder"""
        if self.file_radio.isChecked():
            self.select_files()
        else:
            self.select_folder()

    def select_files(self):
        """Select input files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Media Files", "",
            "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if files:
            self.input_files = files
            self.input_type = 'file'
            self.update_files_info()

    def select_folder(self):
        """Select input folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Media Files")
        if folder:
            self.input_files = self.find_supported_files(folder)
            self.input_type = 'folder'
            self.update_files_info()

    def find_supported_files(self, folder):
        """Find supported media files in folder"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi', '*.mov', '*.mkv']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        return sorted(files)

    def update_files_info(self):
        """Update files information display"""
        if not self.input_files:
            self.files_info_label.setText("No files selected")
            return
            
        count = len(self.input_files)
        if count == 1:
            info = f"1 file selected: {os.path.basename(self.input_files[0])}"
        else:
            info = f"{count} files selected"
            if count <= 10:
                file_list = "\n".join([f"• {os.path.basename(f)}" for f in self.input_files])
                info += f"\n{file_list}"
            else:
                preview = "\n".join([f"• {os.path.basename(f)}" for f in self.input_files[:5]])
                info += f"\n{preview}\n... and {count - 5} more files"
                
        self.files_info_label.setText(info)

    def select_output_folder(self):
        """Select output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(folder)

    def run_analysis(self):
        """Run the analysis"""
        # Validation
        if not self.input_files:
            Notifications.warning(self, "Please select input files or folder first.")
            return
            
        if not self.output_folder:
            Notifications.warning(self, "Please select an output folder.")
            return
            
        project_name = self.project_name_input.text().strip()
        if not project_name:
            Notifications.warning(self, "Please enter a project name.")
            return
            
        # Create project output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = os.path.join(self.output_folder, f"{project_name}_{timestamp}")
        
        # Gather parameters and models based on mode
        params, models_info = self.gather_analysis_parameters()
        
        # Start processing in worker thread
        self.start_analysis_worker(project_folder, params, models_info)

    def gather_analysis_parameters(self):
        """Gather analysis parameters and model info based on current mode"""
        # Shared parameters (available for all modes)
        params = {
            'conf': self.conf_spinbox.value(),
            'device': self.device_combo.currentText() if self.device_combo.currentText() != 'auto' else None,
            'imgsz': int(self.imgsz_combo.currentText()),
        }
        models_info = {}
        
        # Base models directory
        models_base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        if self.analysis_mode == 'detection':
            params.update({
                'iou': self.iou_spinbox.value(),
                'frame_skip': self.frame_skip_spinbox.value(),
                'save_summary': self.save_detection_summary_cb.isChecked(),
                'save_txt': self.save_detection_coords_cb.isChecked(),
                'save_videos_images': self.save_annotated_media_cb.isChecked(),
                'save_crops': self.save_crops_cb.isChecked(),
                'sort_detections': self.sort_detections_cb.isChecked(),
            })
            
            # Construct detection model path
            model_type = self.model_type_combo.currentText()
            if model_type == "Body Detection":
                subdir = "body_detection"
            elif model_type == "Face Detection":
                subdir = "face_detection"
            else:
                subdir = "body_detection"
                
            model_file = self.detection_model_combo.currentText()
            detection_model_path = os.path.join(models_base_dir, subdir, model_file)
            models_info['detection_model'] = detection_model_path
            models_info['model_type'] = model_type
            
        elif self.analysis_mode == 'classification':
            params.update({
                'save_summary': self.save_classification_summary_cb.isChecked(),
            })
            
            # Construct classification model path
            model_file = self.classification_model_combo.currentText()
            classification_model_path = os.path.join(models_base_dir, model_file)
            models_info['classification_model'] = classification_model_path
            
        elif self.analysis_mode == 'detection_classification':
            params.update({
                'iou': self.iou_spinbox.value(),
                'frame_skip': self.frame_skip_spinbox.value(),
                'save_summary': self.save_identity_summary_cb.isChecked(),
                'save_annotated': self.save_annotated_identity_cb.isChecked(),
                'delete_faces': self.delete_faces_cb.isChecked(),
            })
            
            # For identity mode - ALWAYS use face detection models (like identity app)
            subdir = "face_detection"
            detection_model_file = self.detection_model_combo.currentText()
            detection_model_path = os.path.join(models_base_dir, subdir, detection_model_file)
            
            # Identity models are directly in models directory
            identity_model_file = self.classification_model_combo.currentText()
            identity_model_path = os.path.join(models_base_dir, identity_model_file)
            
            models_info['detection_model'] = detection_model_path
            models_info['classification_model'] = identity_model_path
            models_info['model_type'] = "Face Detection"  # Always face detection for identity
            
        return params, models_info

    def start_analysis_worker(self, output_folder, params, models_info):
        """Start the analysis worker thread"""
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")
        
        # Create worker and thread
        self.worker = AnalysisWorker()
        self.thread = threading.Thread(
            target=self.worker.process_files,
            args=(self.input_files, params, output_folder, self.analysis_mode, models_info)
        )
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_completed.connect(self.on_analysis_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        # Start thread
        self.thread.start()

    @pyqtSlot(int, int, str)
    def update_progress(self, current, total, message):
        """Update progress display"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(f"Processing {current}/{total}: {message}")

    @pyqtSlot(str)
    def on_analysis_completed(self, summary):
        """Handle analysis completion"""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Analysis completed successfully!")
        Notifications.success(self, f"Analysis completed!\n{summary}")

    @pyqtSlot(str)
    def on_error_occurred(self, error_msg):
        """Handle analysis error"""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Analysis failed!")
        Notifications.error(self, f"Analysis failed: {error_msg}")

    def load_settings(self):
        """Load saved settings"""
        # Implementation would load saved settings from config
        pass

    def import_detection_model(self):
        """Import a custom detection model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Detection Model", "", 
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Determine target directory based on current model type
            model_type = self.model_type_combo.currentText()
            if model_type == "Body Detection":
                target_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'body_detection')
            else:  # Face Detection
                target_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_detection')
            
            # Create directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the model file
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            
            # Check if file already exists
            if os.path.exists(target_path):
                reply = QMessageBox.question(
                    self, "File Exists", 
                    f"Model '{filename}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            shutil.copy2(file_path, target_path)
            
            # Refresh the model dropdown
            self.update_detection_models()
            
            # Select the imported model
            self.detection_model_combo.setCurrentText(filename)
            
            Notifications.success(self, f"Model '{filename}' imported successfully!")
            
        except Exception as e:
            Notifications.error(self, f"Failed to import model: {str(e)}")

    def import_classification_model(self):
        """Import a custom classification model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Classification Model", "", 
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Target directory is always the main models directory
            target_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            # Create directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the model file
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            
            # Check if file already exists
            if os.path.exists(target_path):
                reply = QMessageBox.question(
                    self, "File Exists", 
                    f"Model '{filename}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            shutil.copy2(file_path, target_path)
            
            # Refresh the model dropdown
            self.update_classification_models()
            
            # Select the imported model
            self.classification_model_combo.setCurrentText(filename)
            
            Notifications.success(self, f"Model '{filename}' imported successfully!")
            
        except Exception as e:
            Notifications.error(self, f"Failed to import model: {str(e)}")

    def save_settings(self):
        """Save current settings"""
        # Implementation would save settings to config
        pass 