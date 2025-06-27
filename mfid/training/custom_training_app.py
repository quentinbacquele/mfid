import sys
import os
import shutil
import threading
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QTabWidget, QFileDialog, QMessageBox, QLineEdit, QCheckBox,
                           QComboBox, QFormLayout, QScrollArea, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QObject, QEvent, pyqtSignal, QPoint, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPen
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit
from mfid.utils.logging_utils import get_logger
from mfid.utils.notifications import Notifications
from ultralytics import YOLO

logger = get_logger('custom_training_app')

# --- ScrollEventFilter for preventing spin boxes from capturing scroll events ---
class ScrollEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            # Ignore the wheel event, let it propagate to the parent (scroll area)
            event.ignore()
            return True # Indicate that the event was handled (by ignoring it)
        # Default handling for other events
        return super().eventFilter(obj, event)

# --- Copied ImageAnnotator from mfid/face/face_annotator.py ---
class ClickableImageLabel(QLabel):
    """Custom QLabel that emits mouse events for bounding box drawing"""
    mouse_pressed = pyqtSignal(QPoint)
    mouse_moved = pyqtSignal(QPoint)
    mouse_released = pyqtSignal(QPoint)
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print(f"ClickableImageLabel: Mouse pressed at {event.pos().x()}, {event.pos().y()}")
            self.mouse_pressed.emit(event.pos())
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_released.emit(event.pos())
        super().mouseReleaseEvent(event)

class ImageAnnotator(QWidget):
    def __init__(self, folderPath):
        super().__init__()
        self.folderPath = folderPath
        # Filter out potential non-image files or subdirectories if any exist
        self.images = sorted([img for img in os.listdir(folderPath)
                             if os.path.isfile(os.path.join(folderPath, img)) and img.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.currentIndex = 0
        self.currentVideo = None # Keeps track of the base name for grouping
        self.annotations = {}  # Dictionary to store annotations {image_filename: label} or {image_filename: {'label': label, 'boxes': [...]}}
        
        # Bounding box functionality
        self.bbox_mode = False
        self.current_image_path = None
        self.current_pixmap = None
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.current_boxes = []  # List of dictionaries: {'label': str, 'x1': int, 'y1': int, 'x2': int, 'y2': int}
        self.drawing_box = False
        self.start_point = None
        self.current_box = None
        self.label_rects = []  # Store label rectangles in original coordinates for click detection: [{'orig_rect': tuple, 'box_index': int}, ...]
        
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        
        # Create a clickable image label for bounding box drawing
        self.image_display = ClickableImageLabel("Load images first", self)
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 300)  # Just set minimum size
        self.image_display.setStyleSheet("""
            QLabel {
                border: 1px solid #555555;
                background-color: #2a2a2a;
            }
        """)
        self.image_display.mouse_pressed.connect(self.start_drawing_box)
        self.image_display.mouse_moved.connect(self.update_drawing_box)
        self.image_display.mouse_released.connect(self.finish_drawing_box)
        
        self.text_input = QLineEdit(self)
        self.next_button = DarkButton('Next Image', self.processAnnotation)
        self.prev_button = DarkButton('Previous Image', self.showPreviousImage)
        self.falsePositiveButton = DarkButton('Mark as False Positive / Delete', self.handleFalsePositive)
        self.retainLabelCheckbox = QCheckBox("Retain Label for Next Image", self)
        
        # Bounding box controls
        self.bbox_checkbox = QCheckBox("Enable Bounding Box Mode", self)
        self.bbox_checkbox.stateChanged.connect(self.toggle_bbox_mode)
        
        self.remove_last_box_button = DarkButton('Remove Last Box', self.remove_last_box)
        self.remove_last_box_button.setEnabled(False)
        
        self.fullscreen_button = DarkButton('🔍 Fullscreen', self.toggle_fullscreen)
        self.fullscreen_button.setToolTip("Toggle fullscreen mode for detailed annotation")
        
        self.boxes_info_label = QLabel("No boxes drawn", self)
        self.boxes_info_label.setStyleSheet("color: #cccccc; font-style: italic;")
        
        # Fullscreen mode tracking
        self.fullscreen_mode = False
        self.normal_layout = None
        self.fullscreen_layout = None

        layout = QVBoxLayout()
        layout.addWidget(self.image_display)
        layout.addWidget(QLabel("Enter Label:"))
        layout.addWidget(self.text_input)

        # Bounding box controls section
        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.bbox_checkbox)
        bbox_controls_layout = QHBoxLayout()
        bbox_controls_layout.addWidget(self.remove_last_box_button)
        bbox_controls_layout.addWidget(self.fullscreen_button)
        bbox_controls_layout.addStretch()
        bbox_layout.addLayout(bbox_controls_layout)
        bbox_layout.addWidget(self.boxes_info_label)
        layout.addLayout(bbox_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.falsePositiveButton)
        layout.addWidget(self.retainLabelCheckbox)

        self.setLayout(layout)
        self.normal_layout = layout  # Store reference to normal layout
        self.setWindowTitle('Image Annotator')
        self.resize(700, 700)  # Initial size, but resizable
        self.showNextImage() # Load first image if available
        
    def resizeEvent(self, event):
        """Handle window resize events by updating the display"""
        super().resizeEvent(event)
        # Update display after resize to recalculate scaling
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            self.update_display()
            
    def toggle_fullscreen(self):
        """Toggle between normal and fullscreen annotation mode"""
        if self.fullscreen_mode:
            self.exit_fullscreen_mode()
        else:
            self.enter_fullscreen_mode()
            
    def enter_fullscreen_mode(self):
        """Enter fullscreen annotation mode"""
        # Set fullscreen mode flag FIRST
        self.fullscreen_mode = True
        
        # Update button text
        self.fullscreen_button.setText('📤 Exit Fullscreen')
        
        # Remove current layout properly
        old_layout = self.layout()
        if old_layout:
            # Remove the layout from the widget but don't delete the widgets
            old_widget = QWidget()
            old_widget.setLayout(old_layout)
        
        # Create fullscreen layout
        # Main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Image takes most of the space (left side)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # Remove size constraints for fullscreen
        self.image_display.setMinimumSize(100, 100)  # Very small minimum
        self.image_display.setMaximumSize(16777215, 16777215)  # Remove max size
        
        image_layout.addWidget(self.image_display)
        
        # Controls panel (right side - compact)
        controls_widget = QWidget()
        controls_widget.setFixedWidth(280)
        controls_widget.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border-left: 2px solid #555555;
            }
        """)
        
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add essential controls to right panel
        controls_layout.addWidget(QLabel("Enter Label:"))
        controls_layout.addWidget(self.text_input)
        
        # Compact button layout
        button_layout1 = QHBoxLayout()
        button_layout1.addWidget(self.prev_button)
        button_layout1.addWidget(self.next_button)
        controls_layout.addLayout(button_layout1)
        
        controls_layout.addWidget(self.falsePositiveButton)
        controls_layout.addWidget(self.retainLabelCheckbox)
        
        # Bounding box controls
        controls_layout.addWidget(self.bbox_checkbox)
        
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.remove_last_box_button)
        button_layout2.addWidget(self.fullscreen_button)
        controls_layout.addLayout(button_layout2)
        
        controls_layout.addWidget(self.boxes_info_label)
        controls_layout.addStretch()
        
        # Add to main layout
        main_layout.addWidget(image_widget, 1)  # Takes most space
        main_layout.addWidget(controls_widget, 0)  # Fixed width
        
        # Store fullscreen layout reference
        self.fullscreen_layout = main_layout
        
        # Expand window first
        self.showMaximized()
        
        # Force a layout update and then update display
        QApplication.processEvents()
        
        # Update display to apply new scaling
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            self.update_display()
        
        logger.info("Entered fullscreen annotation mode")
        
    def clear_layout(self, layout):
        """Recursively clear a layout and all its children"""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
                elif child.layout():
                    self.clear_layout(child.layout())
                    
    def recreate_normal_layout(self):
        """Recreate the normal layout after exiting fullscreen"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        layout.addWidget(self.image_display)
        layout.addWidget(QLabel("Enter Label:"))
        layout.addWidget(self.text_input)

        # Bounding box controls section
        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.bbox_checkbox)
        bbox_controls_layout = QHBoxLayout()
        bbox_controls_layout.addWidget(self.remove_last_box_button)
        bbox_controls_layout.addWidget(self.fullscreen_button)
        bbox_controls_layout.addStretch()
        bbox_layout.addLayout(bbox_controls_layout)
        bbox_layout.addWidget(self.boxes_info_label)
        layout.addLayout(bbox_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.falsePositiveButton)
        layout.addWidget(self.retainLabelCheckbox)

        self.normal_layout = layout
                    
    def exit_fullscreen_mode(self):
        """Exit fullscreen annotation mode"""
        # Set fullscreen mode flag to False FIRST
        self.fullscreen_mode = False
        
        # Update button text
        self.fullscreen_button.setText('🔍 Fullscreen')
        
        # Restore normal image display constraints
        self.image_display.setMinimumSize(400, 300)
        self.image_display.setMaximumSize(16777215, 16777215)
        
        # Remove current layout properly
        current_layout = self.layout()
        if current_layout:
            # Remove the layout from the widget but don't delete the widgets
            old_widget = QWidget()
            old_widget.setLayout(current_layout)
        
        # Recreate and restore normal layout
        self.recreate_normal_layout()
        
        # Restore normal window size first
        self.showNormal()
        self.resize(700, 700)
        
        # Force a layout update and then update display
        QApplication.processEvents()
        
        # Update display to apply new scaling
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            self.update_display()
        
        logger.info("Exited fullscreen annotation mode")
        
    def toggle_bbox_mode(self, state):
        """Toggle bounding box annotation mode"""
        self.bbox_mode = state == 2  # 2 means checked
        logger.info(f"Bbox mode toggled to: {self.bbox_mode}")
        self.update_remove_button_state()
        
        if self.bbox_mode:
            self.image_display.setCursor(Qt.CrossCursor)
            self.boxes_info_label.setText("Click and drag to draw bounding boxes")
            logger.info("Bbox mode enabled - cursor set to cross")
        else:
            self.image_display.setCursor(Qt.ArrowCursor)
            self.boxes_info_label.setText("Bounding box mode disabled")
            self.current_boxes = []  # Clear boxes when disabling
            logger.info("Bbox mode disabled - boxes cleared")
        
        self.update_display()
        
    def widget_to_image_coords(self, widget_point):
        """Convert widget coordinates to image coordinates within the label"""
        if not self.image_display.pixmap():
            return widget_point
            
        # Get the label and pixmap dimensions
        label_size = self.image_display.size()
        pixmap_size = self.image_display.pixmap().size()
        
        # Calculate the position of the image within the label (centered)
        x_offset = max(0, (label_size.width() - pixmap_size.width()) // 2)
        y_offset = max(0, (label_size.height() - pixmap_size.height()) // 2)
        
        # Adjust coordinates to be relative to the image
        image_x = widget_point.x() - x_offset
        image_y = widget_point.y() - y_offset
        
        # Clamp to image bounds (ensure we're within the displayed image)
        image_x = max(0, min(image_x, pixmap_size.width() - 1))
        image_y = max(0, min(image_y, pixmap_size.height() - 1))
        
        return QPoint(image_x, image_y)

    def start_drawing_box(self, point):
        """Start drawing a bounding box or edit existing label"""
        logger.info(f"Mouse pressed at {point.x()}, {point.y()}, bbox_mode: {self.bbox_mode}, has_pixmap: {self.current_pixmap is not None}")
        
        if not self.bbox_mode:
            logger.info("Bbox mode not enabled")
            return
            
        if not self.current_pixmap:
            logger.info("No current pixmap available")
            return
            
        # Check if clicking on an existing label
        clicked_label = self.get_clicked_label(point)
        if clicked_label is not None:
            logger.info(f"Clicked on label {clicked_label}: '{self.current_boxes[clicked_label]['label']}'")
            self.edit_label_dialog(clicked_label)
            return
            
        # Convert widget coordinates to image coordinates
        image_point = self.widget_to_image_coords(point)
        logger.info(f"Starting new box at image coords: {image_point.x()}, {image_point.y()}")
        
        self.drawing_box = True
        self.start_point = image_point
        self.current_box = None
        
    def update_drawing_box(self, point):
        """Update the bounding box being drawn"""
        if not self.bbox_mode or not self.drawing_box or not self.start_point:
            return
            
        # Convert widget coordinates to image coordinates
        image_point = self.widget_to_image_coords(point)
            
        # Create temporary box for preview
        x1 = min(self.start_point.x(), image_point.x())
        y1 = min(self.start_point.y(), image_point.y())
        x2 = max(self.start_point.x(), image_point.x())
        y2 = max(self.start_point.y(), image_point.y())
        
        self.current_box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        self.update_display()
        
    def finish_drawing_box(self, point):
        """Finish drawing a bounding box and save it"""
        if not self.bbox_mode or not self.drawing_box or not self.start_point:
            return
            
        self.drawing_box = False
        
        # Convert widget coordinates to image coordinates
        image_point = self.widget_to_image_coords(point)
        
        # Calculate final box coordinates
        x1 = min(self.start_point.x(), image_point.x())
        y1 = min(self.start_point.y(), image_point.y())
        x2 = max(self.start_point.x(), image_point.x())
        y2 = max(self.start_point.y(), image_point.y())
        
        # Only save box if it has minimum size
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            # Convert from display coordinates to original image coordinates
            original_coords = self.display_to_original_coords(x1, y1, x2, y2)
            
            label = self.text_input.text().strip()
            if not label:
                label = "unlabeled"
                
            box_data = {
                'label': label,
                'x1': original_coords[0],
                'y1': original_coords[1], 
                'x2': original_coords[2],
                'y2': original_coords[3]
            }
            
            self.current_boxes.append(box_data)
            self.update_boxes_info()
            self.update_remove_button_state()
            
        self.current_box = None
        self.update_display()
        
    def display_to_original_coords(self, x1, y1, x2, y2):
        """Convert display coordinates to original image coordinates"""
        if not self.original_pixmap:
            return (x1, y1, x2, y2)
            
        # Get the actual display size of the scaled pixmap
        display_size = self.image_display.pixmap().size()
        original_size = self.original_pixmap.size()
        
        # Calculate scaling factors
        scale_x = original_size.width() / display_size.width()
        scale_y = original_size.height() / display_size.height()
        
        # Convert coordinates
        orig_x1 = int(x1 * scale_x)
        orig_y1 = int(y1 * scale_y) 
        orig_x2 = int(x2 * scale_x)
        orig_y2 = int(y2 * scale_y)
        
        return (orig_x1, orig_y1, orig_x2, orig_y2)
        
    def original_to_display_coords(self, x1, y1, x2, y2):
        """Convert original image coordinates to display coordinates"""
        if not self.original_pixmap:
            return (x1, y1, x2, y2)
            
        # Get the actual display size of the scaled pixmap
        display_size = self.image_display.pixmap().size()
        original_size = self.original_pixmap.size()
        
        # Calculate scaling factors
        scale_x = display_size.width() / original_size.width()
        scale_y = display_size.height() / original_size.height()
        
        # Convert coordinates
        disp_x1 = int(x1 * scale_x)
        disp_y1 = int(y1 * scale_y)
        disp_x2 = int(x2 * scale_x)
        disp_y2 = int(y2 * scale_y)
        
        return (disp_x1, disp_y1, disp_x2, disp_y2)
        
    def remove_last_box(self):
        """Remove the last bounding box created for the current image"""
        if self.current_boxes:
            removed_box = self.current_boxes.pop()
            logger.info(f"Removed last bounding box: {removed_box['label']}")
        self.update_boxes_info()
        self.update_remove_button_state()
        self.update_display()
        
    def update_remove_button_state(self):
        """Update the remove button enabled state based on available boxes and mode"""
        has_boxes = len(self.current_boxes) > 0
        self.remove_last_box_button.setEnabled(self.bbox_mode and has_boxes)
        
    def get_clicked_label(self, widget_point):
        """Check if a label was clicked and return its index"""
        if not self.original_pixmap or not self.label_rects:
            return None
            
        # Convert widget click point to image coordinates
        image_point = self.widget_to_image_coords(widget_point)
        
        # Convert to original image coordinates
        if not self.image_display.pixmap():
            return None
            
        display_size = self.image_display.pixmap().size()
        original_size = self.original_pixmap.size()
        
        # Calculate scaling factors
        scale_x = original_size.width() / display_size.width()
        scale_y = original_size.height() / display_size.height()
        
        # Convert image point to original coordinates
        orig_x = int(image_point.x() * scale_x)
        orig_y = int(image_point.y() * scale_y)
        
        # Check which label rectangle contains this point
        for label_info in self.label_rects:
            orig_rect = label_info['orig_rect']
            if (orig_rect[0] <= orig_x <= orig_rect[0] + orig_rect[2] and 
                orig_rect[1] <= orig_y <= orig_rect[1] + orig_rect[3]):
                return label_info['box_index']
        return None
        
    def edit_label_dialog(self, box_index):
        """Open dialog to edit or remove a bounding box label"""
        if box_index >= len(self.current_boxes):
            return
            
        box = self.current_boxes[box_index]
        current_label = box['label']
        
        # Create custom dialog
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Bounding Box")
        dialog.resize(300, 120)
        
        # Apply dark theme
        DarkTheme(dialog)
        
        layout = QVBoxLayout(dialog)
        
        # Label input
        layout.addWidget(QLabel("Label:"))
        label_input = QLineEdit(current_label)
        DarkLineEdit(label_input, "Enter label")
        layout.addWidget(label_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_button = DarkButton("Apply", lambda: self.apply_label_edit(dialog, box_index, label_input.text()))
        remove_button = DarkButton("Remove Box", lambda: self.remove_specific_box(dialog, box_index))
        cancel_button = DarkButton("Cancel", dialog.reject)
        
        # Style the remove button with red color
        remove_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c62828;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        
        button_layout.addWidget(apply_button)
        button_layout.addWidget(remove_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Set default button and focus
        apply_button.setDefault(True)
        label_input.setFocus()
        label_input.selectAll()
        
        dialog.exec_()
        
    def apply_label_edit(self, dialog, box_index, new_label):
        """Apply label changes to a bounding box"""
        if box_index < len(self.current_boxes):
            old_label = self.current_boxes[box_index]['label']
            self.current_boxes[box_index]['label'] = new_label.strip() or 'unlabeled'
            logger.info(f"Updated bounding box label: '{old_label}' → '{self.current_boxes[box_index]['label']}'")
            
            self.update_boxes_info()
            self.update_display()
            dialog.accept()
    
    def remove_specific_box(self, dialog, box_index):
        """Remove a specific bounding box"""
        if box_index < len(self.current_boxes):
            removed_box = self.current_boxes.pop(box_index)
            logger.info(f"Removed bounding box: '{removed_box['label']}' at ({removed_box['x1']}, {removed_box['y1']})")
            
            self.update_boxes_info()
            self.update_remove_button_state()
            self.update_display()
            dialog.accept()
        
    def update_boxes_info(self):
        """Update the box information display"""
        if not self.current_boxes:
            self.boxes_info_label.setText("No boxes drawn")
        else:
            box_count = len(self.current_boxes)
            labels = [box['label'] for box in self.current_boxes]
            unique_labels = list(set(labels))
            self.boxes_info_label.setText(f"{box_count} boxes: {', '.join(unique_labels)}")
            
    def update_display(self):
        """Update the image display with bounding boxes"""
        if not self.original_pixmap:
            return
            
        # Scale image to fit within the current widget size, maintaining aspect ratio
        widget_size = self.image_display.size()
        available_width = widget_size.width() - 20  # Leave some margin
        available_height = widget_size.height() - 20
        
        # In fullscreen mode, allow maximum scaling. In normal mode, ensure minimum size.
        if hasattr(self, 'fullscreen_mode') and self.fullscreen_mode:
            # Fullscreen: use almost all available space for maximum detail
            available_width = max(200, available_width)  # Just prevent extreme cases
            available_height = max(150, available_height)
        else:
            # Normal mode: ensure reasonable minimum size
            available_width = max(400, available_width)
            available_height = max(300, available_height)
        
        pixmap = self.original_pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.current_pixmap = pixmap  # Store the scaled pixmap
        
        # Clear label rectangles for click detection
        self.label_rects = []
        
        if self.bbox_mode and (self.current_boxes or self.current_box):
            # Draw on a copy
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw existing boxes
            for box_index, box in enumerate(self.current_boxes):
                x1, y1, x2, y2 = self.original_to_display_coords(
                    box['x1'], box['y1'], box['x2'], box['y2']
                )
                
                # Draw box
                pen = QPen(Qt.red, 2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                # Draw label with black background
                label_text = box['label']
                font = QFont('Arial', 14, QFont.Bold)  # Bigger and bold font
                painter.setFont(font)
                
                # Calculate text size for background
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(label_text)
                text_height = font_metrics.height()
                
                # Position label above the box, but ensure it's visible
                label_x = x1
                label_y = max(y1 - 8, text_height)  # Ensure label is not cut off at top
                
                # Draw black background rectangle
                background_padding = 4
                background_rect = painter.fillRect(
                    label_x - background_padding, 
                    label_y - text_height, 
                    text_width + 2 * background_padding, 
                    text_height + background_padding,
                    Qt.black
                )
                
                # Store label rectangle in original coordinates for click detection
                # Convert display label position back to original coordinates
                orig_label_coords = self.display_to_original_coords(
                    label_x - background_padding, 
                    label_y - text_height,
                    label_x + text_width + background_padding,
                    label_y + background_padding
                )
                
                # Store as (x, y, width, height) in original coordinates
                orig_label_rect = (
                    orig_label_coords[0],  # x
                    orig_label_coords[1],  # y
                    orig_label_coords[2] - orig_label_coords[0],  # width
                    orig_label_coords[3] - orig_label_coords[1]   # height
                )
                self.label_rects.append({'orig_rect': orig_label_rect, 'box_index': box_index})
                
                # Draw white text on black background
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(label_x, label_y - 2, label_text)
                
            # Draw current box being drawn
            if self.current_box:
                pen = QPen(Qt.yellow, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(
                    self.current_box['x1'], 
                    self.current_box['y1'],
                    self.current_box['x2'] - self.current_box['x1'],
                    self.current_box['y2'] - self.current_box['y1']
                )
                
            painter.end()
            
        self.image_display.setPixmap(pixmap)

    def processAnnotation(self):
        if self.currentIndex >= len(self.images):
            QMessageBox.information(self, 'Done', 'All images processed.')
            return

        current_image = self.images[self.currentIndex]
        annotation = self.text_input.text().strip()
        
        if self.bbox_mode:
            # Save bounding box annotation
            if self.current_boxes or annotation:
                annotation_data = {
                    'label': annotation if annotation else 'unlabeled',
                    'boxes': self.current_boxes.copy()
                }
                self.annotations[current_image] = annotation_data
                
                # Save coordinates to file in the same format as the identity app
                self.save_bbox_coordinates(current_image, annotation_data)
            elif current_image in self.annotations:
                # Remove annotation if no boxes and no label
                del self.annotations[current_image]
                self.delete_bbox_coordinates_file(current_image)
        else:
            # Save regular annotation
            if annotation:
                self.annotations[current_image] = annotation
            elif current_image in self.annotations:
                # Remove annotation if cleared
                del self.annotations[current_image]

        self.currentIndex += 1
        self.showNextImage()
        
    def save_bbox_coordinates(self, image_filename, annotation_data):
        """Save bounding box coordinates to file in the same format as identity app"""
        if not annotation_data['boxes']:
            return
            
        # Create coordinates folder in the root annotation folder (not in subfolders)
        coords_folder = os.path.join(self.folderPath, 'coordinates')
        os.makedirs(coords_folder, exist_ok=True)
        
        # Create filename for coordinates
        base_name = os.path.splitext(image_filename)[0]
        coords_filename = f"{base_name}_coordinates.txt"
        coords_path = os.path.join(coords_folder, coords_filename)
        
        # Write coordinates in the same format as the identity app
        with open(coords_path, 'w') as f:
            f.write(f"Image: {image_filename}\n")
            for box in annotation_data['boxes']:
                # Format: label confidence x1 y1 x2 y2
                # We use confidence = 1.0 for manual annotations
                f.write(f"{box['label']} 1.0000 {box['x1']} {box['y1']} {box['x2']} {box['y2']}\n")
        
        logger.info(f"Saved bounding box coordinates for {image_filename} to {coords_path}")
        
    def delete_bbox_coordinates_file(self, image_filename):
        """Delete the coordinates file for an image"""
        coords_folder = os.path.join(self.folderPath, 'coordinates')
        base_name = os.path.splitext(image_filename)[0]
        coords_filename = f"{base_name}_coordinates.txt"
        coords_path = os.path.join(coords_folder, coords_filename)
        
        if os.path.exists(coords_path):
            try:
                os.remove(coords_path)
                logger.info(f"Deleted coordinates file: {coords_path}")
            except Exception as e:
                logger.error(f"Error deleting coordinates file {coords_path}: {e}")

    def showPreviousImage(self):
        if self.currentIndex > 0:
            self.currentIndex -= 1
            self.showNextImage()

    def showNextImage(self):
        if self.currentIndex < len(self.images):
            current_image = self.images[self.currentIndex]
            image_path = os.path.join(self.folderPath, current_image)

            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                logger.error(f"Failed to load image: {image_path}")
                # Skip corrupted/unreadable image
                self.images.pop(self.currentIndex)
                if self.currentIndex >= len(self.images):
                    QMessageBox.information(self, 'Done', 'Finished annotating images.')
                    self.finalizeAnnotations()
                    self.close()
                    return
                self.showNextImage() # Try next one
                return

            # Store original pixmap and current image path for bounding box functionality
            self.original_pixmap = pixmap
            self.current_image_path = image_path
            self.current_pixmap = None  # Will be set in update_display()
            
            # Load existing bounding boxes for this image
            self.load_current_image_boxes(current_image)
            
            self.update_display()
            self.setWindowTitle(f'Image Annotator - ({self.currentIndex + 1}/{len(self.images)}) {current_image}')

            # Update text input based on stored annotation or retain checkbox
            if current_image in self.annotations:
                if isinstance(self.annotations[current_image], dict):
                    # Bounding box annotation
                    self.text_input.setText(self.annotations[current_image].get('label', ''))
                else:
                    # Simple string annotation
                    self.text_input.setText(self.annotations[current_image])
            elif not self.retainLabelCheckbox.isChecked():
                self.text_input.clear()
        else:
            QMessageBox.information(self, 'Done', 'All images have been annotated.')
            self.finalizeAnnotations()
            self.close()
            
    def load_current_image_boxes(self, image_filename):
        """Load existing bounding boxes for the current image"""
        self.current_boxes = []
        
        if image_filename in self.annotations and isinstance(self.annotations[image_filename], dict):
            self.current_boxes = self.annotations[image_filename].get('boxes', []).copy()
            
        self.update_boxes_info()
        self.update_remove_button_state()
        
    def get_primary_label_from_bbox_annotation(self, annotation_data):
        """
        Determine the primary label for an image with bounding box annotations.
        Uses the most common label among all bounding boxes, or the overall image label if available.
        """
        # First check if there's an overall image label
        if annotation_data.get('label') and annotation_data['label'] != 'unlabeled':
            return annotation_data['label']
        
        # If no overall label, find the most common label among bounding boxes
        boxes = annotation_data.get('boxes', [])
        if not boxes:
            return None
            
        # Count label occurrences
        label_counts = {}
        for box in boxes:
            label = box.get('label', 'unlabeled')
            if label != 'unlabeled':  # Skip unlabeled boxes
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if not label_counts:
            # All boxes are unlabeled, return the first non-unlabeled one or None
            for box in boxes:
                if box.get('label', 'unlabeled') != 'unlabeled':
                    return box['label']
            return None
        
        # Return the most frequent label
        primary_label = max(label_counts, key=label_counts.get)
        logger.info(f"Primary label determined: '{primary_label}' (appears {label_counts[primary_label]} times)")
        return primary_label

    def handleFalsePositive(self):
        if self.currentIndex < len(self.images):
            image_to_delete = self.images[self.currentIndex]
            delete_path = os.path.join(self.folderPath, image_to_delete)
            reply = QMessageBox.question(self, 'Confirm Deletion',
                                       f"Are you sure you want to delete this image?\n{image_to_delete}",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                try:
                    os.remove(delete_path)
                    logger.info(f"Deleted image: {delete_path}")
                    # Remove from annotations if present
                    if image_to_delete in self.annotations:
                        del self.annotations[image_to_delete]
                    # Remove from list and show next
                    self.images.pop(self.currentIndex)
                    # No need to increment currentIndex as pop shifts list
                    self.showNextImage()
                except OSError as e:
                    logger.error(f"Error deleting image {delete_path}: {e}")
                    QMessageBox.warning(self, 'Deletion Error', f'Could not delete image: {e}')
            else:
                # User cancelled deletion
                pass
        else:
            QMessageBox.information(self, 'Done', 'All images processed.')

    def finalizeAnnotations(self):
        logger.info(f"Finalizing annotations for folder: {self.folderPath}")
        annotated_count = 0
        bbox_count = 0
        
        for image_filename, annotation_data in self.annotations.items():
            if isinstance(annotation_data, dict):
                # Bounding box annotation - save coordinates AND move image to primary label folder
                bbox_count += 1
                logger.debug(f"Bounding box annotation saved for {image_filename}")
                
                # Determine primary label for classification folder
                primary_label = self.get_primary_label_from_bbox_annotation(annotation_data)
                
                if primary_label:
                    # Move image to primary label subfolder (for classification)
                    annotation_folder = os.path.join(self.folderPath, primary_label)
                    os.makedirs(annotation_folder, exist_ok=True)

                    source_path = os.path.join(self.folderPath, image_filename)
                    destination_path = os.path.join(annotation_folder, image_filename)

                    if os.path.exists(source_path):
                        try:
                            shutil.move(source_path, destination_path)
                            logger.debug(f"Moved bounding box annotated image {source_path} to {destination_path}")
                            annotated_count += 1
                        except Exception as e:
                            logger.error(f"Error moving {source_path} to {destination_path}: {e}")
                            QMessageBox.warning(self, 'Move Error', f'Could not move {image_filename}: {e}')
                    else:
                        logger.warning(f"Source image {source_path} not found during finalization.")
            else:
                # Regular annotation - move image to labeled folder
                annotation_label = annotation_data
                annotation_folder = os.path.join(self.folderPath, annotation_label)
                os.makedirs(annotation_folder, exist_ok=True)

                source_path = os.path.join(self.folderPath, image_filename)
                destination_path = os.path.join(annotation_folder, image_filename) # Keep original name

                if os.path.exists(source_path):
                    try:
                        shutil.move(source_path, destination_path)
                        logger.debug(f"Moved {source_path} to {destination_path}")
                        annotated_count += 1
                    except Exception as e:
                        logger.error(f"Error moving {source_path} to {destination_path}: {e}")
                        QMessageBox.warning(self, 'Move Error', f'Could not move {image_filename}: {e}')
                else:
                    logger.warning(f"Source image {source_path} not found during finalization.")
        
        # Show completion message
        message_parts = []
        if annotated_count > 0:
            message_parts.append(f'{annotated_count} images moved to respective class folders')
        if bbox_count > 0:
            message_parts.append(f'{bbox_count} images annotated with bounding boxes (coordinates saved + images organized by primary label)')
            
        if message_parts:
            QMessageBox.information(self, 'Annotation Complete', '. '.join(message_parts) + '.')
            
        logger.info("Annotation finalization finished.")

    def closeEvent(self, event):
        # Ask user if they want to finalize before closing if there are pending annotations
        if self.annotations and self.currentIndex < len(self.images):
            reply = QMessageBox.question(self, 'Finalize Annotations',
                                      "Do you want to save annotations by moving images to label folders before closing?",
                                      QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.finalizeAnnotations()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else: # Cancel
                event.ignore()
        else:
            self.finalizeAnnotations() # Finalize if done or no annotations made
            event.accept()
# --- End of Copied ImageAnnotator ---

class CustomTrainingApp(QWidget):
    # Training progress signals
    training_progress_updated = pyqtSignal(int, int, str)  # current_epoch, total_epochs, status
    training_metrics_updated = pyqtSignal(str)  # metrics_text
    
    def __init__(self):
        super().__init__()
        self.dataFolder = None # Common variable for the loaded data folder
        # Set default output folder to mfid directory
        self.outputFolder = os.path.join(os.path.dirname(__file__), '..', '..')
        self.outputFolder = os.path.abspath(self.outputFolder)  # Get absolute path
        self.annotator_instance = None # To hold the annotator window
        self.initUI()
        
    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Custom Model Training & Annotation')
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
        title_label = QLabel("Custom Training & Annotation")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        description_label = QLabel("Annotate images and train custom YOLOv8 classification or detection models")
        description_label.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        description_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        main_layout.addLayout(header_layout)
        
        # Data folder selection - Step 1
        folder_group = QGroupBox("Step 1: Select Data Folder")
        folder_group.setStyleSheet("""
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
        folder_layout = QVBoxLayout(folder_group)
        
        folder_description = QLabel("Select a folder containing images for annotation or training.")
        folder_description.setStyleSheet("font-style: italic; color: #aaaaaa; margin-bottom: 10px;")
        folder_layout.addWidget(folder_description)
        
        folder_select_layout = QHBoxLayout()
        self.loadFolderButton = DarkButton('Browse...', self.openDataFolderDialog)
        folder_select_layout.addWidget(QLabel("Data Folder:"))
        folder_select_layout.addWidget(self.loadFolderButton)
        folder_select_layout.addStretch()
        folder_layout.addLayout(folder_select_layout)
        
        self.folder_label = QLabel("Not Selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        folder_layout.addWidget(self.folder_label)
        
        # Add output folder selector
        output_select_layout = QHBoxLayout()
        self.outputFolderButton = DarkButton('Browse...', self.selectOutputFolder)
        output_select_layout.addWidget(QLabel("Output Folder:"))
        output_select_layout.addWidget(self.outputFolderButton)
        output_select_layout.addStretch()
        folder_layout.addLayout(output_select_layout)
        
        self.outputFolder_label = QLabel(self.outputFolder)
        self.outputFolder_label.setWordWrap(True)
        self.outputFolder_label.setStyleSheet("font-style: italic; color: #cccccc; padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        folder_layout.addWidget(self.outputFolder_label)
        
        # Project name
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Project Name:"))
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("e.g., monkey_classification_2024")
        DarkLineEdit(self.project_name_input, "e.g., monkey_classification_2024")
        project_layout.addWidget(self.project_name_input)
        folder_layout.addLayout(project_layout)
        
        main_layout.addWidget(folder_group)
        
        # Tab Widget for Annotation and Training - Step 2
        # Add a group box around the tabs
        tabs_group = QGroupBox("Step 2: Annotate and Train")
        tabs_group.setStyleSheet("""
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
        tabs_layout = QVBoxLayout(tabs_group)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2d2d2d;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #333333;
                color: #bbbbbb;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #ffffff;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3c3c3c;
            }
        """)
        tabs_layout.addWidget(self.tabs)
        
        # --- Annotation Tab ---
        self.annotation_tab = QWidget()
        self.tabs.addTab(self.annotation_tab, "Image Annotation")
        annotation_layout = QVBoxLayout(self.annotation_tab)
        
        # Annotation instructions
        annotation_instructions = QLabel(
            "Use the annotation tools to organize images into class folders for training.\n"
            "1. Manual annotation allows you to view and label each image individually.\n"
            "2. Auto-grouping moves images with specific keywords in their filename to named subfolders."
        )
        annotation_instructions.setStyleSheet("color: #aaaaaa; font-style: italic; margin-bottom: 15px;")
        annotation_instructions.setWordWrap(True)
        annotation_layout.addWidget(annotation_instructions)
        
        # Manual annotation section
        manual_group = QGroupBox("Manual Annotation")
        manual_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        manual_layout = QVBoxLayout(manual_group)
        
        self.annotateButton = DarkButton('Launch Manual Annotator', self.launchAnnotator)
        self.annotateButton.setToolTip("Manually label images in the loaded folder. Labeled images will be moved to subfolders.")
        manual_layout.addWidget(self.annotateButton)
        
        annotation_layout.addWidget(manual_group)
        
        # Auto grouping section
        auto_group = QGroupBox("Auto-Grouping by Keyword")
        auto_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        auto_layout = QVBoxLayout(auto_group)
        
        auto_layout.addWidget(QLabel("Enter a keyword to find in image filenames:"))
        self.autoAnnotateInput = QLineEdit(self)
        DarkLineEdit(self.autoAnnotateInput, 'e.g., macaque, baboon, etc.')
        auto_layout.addWidget(self.autoAnnotateInput)
        
        self.autoAnnotateButton = DarkButton('Auto-Group by Keyword', self.autoAnnotate)
        self.autoAnnotateButton.setToolTip("Move images containing the keyword in their filename to a subfolder named after the keyword.")
        auto_layout.addWidget(self.autoAnnotateButton)
        
        annotation_layout.addWidget(auto_group)
        annotation_layout.addStretch()
        
        # --- Training Tab ---
        self.training_tab = QWidget()
        self.tabs.addTab(self.training_tab, "YOLOv8 Training")
        training_layout = QVBoxLayout(self.training_tab)
        
        # Training mode selection
        mode_group = QGroupBox("Training Mode")
        mode_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        mode_layout = QVBoxLayout(mode_group)
        
        self.training_mode_combo = QComboBox(self)
        self.training_mode_combo.addItems(['Classification', 'Detection'])
        self.training_mode_combo.setCurrentText('Classification')
        self.training_mode_combo.setToolTip("Choose between classification (entire image) or detection (objects in image)")
        self.training_mode_combo.currentTextChanged.connect(self.update_training_instructions)
        mode_layout.addWidget(self.training_mode_combo)
        
        training_layout.addWidget(mode_group)
        
        # Training instructions (will be updated based on mode)
        self.training_instructions = QLabel("")
        self.training_instructions.setStyleSheet("color: #aaaaaa; font-style: italic; margin-bottom: 15px;")
        self.training_instructions.setWordWrap(True)
        training_layout.addWidget(self.training_instructions)
        
        # Initialize instructions
        self.update_training_instructions('Classification')
        
        # Basic configuration
        basic_config_group = QGroupBox("Basic Configuration")
        basic_config_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        
        basic_form_layout = QFormLayout(basic_config_group)
        
        # Apply scroll event filter to combo boxes and spin boxes
        scroll_filter = ScrollEventFilter(self)
        
        # Model size selector
        self.modelSizeComboBox = QComboBox(self)
        self.modelSizeComboBox.addItems(['n', 's', 'm', 'l', 'x'])
        self.modelSizeComboBox.setCurrentText('m')  # Default to medium
        self.modelSizeComboBox.setToolTip("Model size: n=smallest, x=largest (affects accuracy and speed)")
        self.modelSizeComboBox.installEventFilter(scroll_filter)
        basic_form_layout.addRow("Model Size:", self.modelSizeComboBox)
        
        # Epochs input
        self.epochsLineEdit = QLineEdit(self)
        self.epochsLineEdit.setPlaceholderText('e.g., 100')
        self.epochsLineEdit.setText('100')  # Default epochs
        self.epochsLineEdit.setToolTip("Number of complete passes through the training dataset")
        DarkLineEdit(self.epochsLineEdit, 'e.g., 100')
        basic_form_layout.addRow("Epochs:", self.epochsLineEdit)
        
        # Image size selection
        self.imgSizeComboBox = QComboBox(self)
        self.imgSizeComboBox.addItems(['320', '416', '512', '640', '768', '1024'])
        self.imgSizeComboBox.setCurrentText('640')  # Default image size
        self.imgSizeComboBox.setToolTip("Input image size for training (larger = more accurate but slower)")
        self.imgSizeComboBox.installEventFilter(scroll_filter)
        basic_form_layout.addRow("Image Size:", self.imgSizeComboBox)
        
        # Batch size
        self.batchSizeLineEdit = QLineEdit(self)
        self.batchSizeLineEdit.setText('32')  # Default batch size
        self.batchSizeLineEdit.setToolTip("Number of images processed at once (lower for less memory usage)")
        DarkLineEdit(self.batchSizeLineEdit, 'e.g., 32')
        basic_form_layout.addRow("Batch Size:", self.batchSizeLineEdit)
        
        # Note: Project name is now defined in Step 1 at the top
        
        # Device selection
        self.deviceComboBox = QComboBox(self)
        self.deviceComboBox.addItems(['cpu', 'mps', 'cuda:0'])  # Updated device options to include mps
        self.deviceComboBox.setCurrentText('mps')  # Set mps as default for Apple Silicon
        self.deviceComboBox.setToolTip("Processing device (cpu=CPU only, mps=Apple GPU, cuda:0=NVIDIA GPU)")
        self.deviceComboBox.installEventFilter(scroll_filter)
        basic_form_layout.addRow("Device:", self.deviceComboBox)
        
        training_layout.addWidget(basic_config_group)
        
        # Advanced configuration group
        advanced_config_group = QGroupBox("Advanced Options")
        advanced_config_group.setObjectName("Advanced Options")
        advanced_config_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        advanced_config_group.setCheckable(True)
        advanced_config_group.setChecked(False)  # Collapsed by default
        
        advanced_form_layout = QFormLayout(advanced_config_group)
        
        # Learning rate
        self.learningRateLineEdit = QLineEdit(self)
        self.learningRateLineEdit.setText('0.001')
        self.learningRateLineEdit.setToolTip("Initial learning rate (smaller values for fine-tuning)")
        DarkLineEdit(self.learningRateLineEdit, 'e.g., 0.001')
        advanced_form_layout.addRow("Learning Rate:", self.learningRateLineEdit)
        
        # Patience for early stopping
        self.patienceLineEdit = QLineEdit(self)
        self.patienceLineEdit.setText('20')
        self.patienceLineEdit.setToolTip("Epochs to wait for improvement before early stopping")
        DarkLineEdit(self.patienceLineEdit, 'e.g., 20')
        advanced_form_layout.addRow("Patience:", self.patienceLineEdit)
        
        # Augmentation options
        self.augCheckBox = QCheckBox("Enable Data Augmentation")
        self.augCheckBox.setChecked(True)
        self.augCheckBox.setToolTip("Apply random transforms to training images to improve model generalization")
        advanced_form_layout.addRow("", self.augCheckBox)
        
        # Optimizer
        self.optimizerComboBox = QComboBox(self)
        self.optimizerComboBox.addItems(['AdamW', 'SGD', 'Adam', 'RMSProp'])
        self.optimizerComboBox.setCurrentText('AdamW')
        self.optimizerComboBox.setToolTip("Optimization algorithm for training")
        self.optimizerComboBox.installEventFilter(scroll_filter)
        advanced_form_layout.addRow("Optimizer:", self.optimizerComboBox)
        
        # Save best models only
        self.saveBestCheckBox = QCheckBox("Save Best Model Only")
        self.saveBestCheckBox.setChecked(True)
        self.saveBestCheckBox.setToolTip("Only save the best model instead of all checkpoints")
        advanced_form_layout.addRow("", self.saveBestCheckBox)
        
        training_layout.addWidget(advanced_config_group)
        
        # Training controls
        controls_layout = QVBoxLayout()
        self.trainingButton = DarkButton('Start Training', self.runTraining)
        self.trainingButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        controls_layout.addWidget(self.trainingButton)
        
        # Status box with progress info
        status_group = QGroupBox("Training Status")
        status_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        status_layout = QVBoxLayout(status_group)
        
        self.trainingStatusLabel = QLabel('Status: Ready')
        self.trainingStatusLabel.setStyleSheet("color: #cccccc; padding: 5px;")
        status_layout.addWidget(self.trainingStatusLabel)
        
        # Progress bar
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setVisible(False)
        self.training_progress_bar.setMinimum(0)
        self.training_progress_bar.setMaximum(100)
        self.training_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2a2a2a;
                height: 25px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.training_progress_bar)
        
        # Epoch counter display
        self.epoch_counter_label = QLabel("")
        self.epoch_counter_label.setStyleSheet("""
            color: #2196F3; 
            font-weight: bold; 
            font-size: 16px; 
            padding: 5px;
            border: 1px solid #444444;
            border-radius: 3px;
            background-color: #2a2a2a;
        """)
        self.epoch_counter_label.setVisible(False)
        self.epoch_counter_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.epoch_counter_label)
        
        # Training time display
        self.training_time_label = QLabel("")
        self.training_time_label.setStyleSheet("color: #888888; font-size: 12px; padding: 5px; text-align: center;")
        self.training_time_label.setVisible(False)
        self.training_time_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.training_time_label)
        
        # Live metrics display
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("""
            color: #aaaaaa; 
            font-size: 12px; 
            padding: 8px; 
            font-family: monospace;
            background-color: #1e1e1e;
            border: 1px solid #333333;
            border-radius: 3px;
        """)
        self.metrics_label.setVisible(False)
        self.metrics_label.setWordWrap(True)
        status_layout.addWidget(self.metrics_label)
        
        training_layout.addLayout(controls_layout)
        training_layout.addWidget(status_group)
        training_layout.addStretch()
        
        main_layout.addWidget(tabs_group)
        
        # Set the scroll area's widget to our container
        scroll_area.setWidget(scroll_content)
        
        # Create the main window layout and add the scroll area
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll_area)
        
        # Set a reasonable fixed size for the window
        self.setFixedSize(800, 700)
        
        # Create output folder if it doesn't exist
        os.makedirs(self.outputFolder, exist_ok=True)
        
        # Connect progress signals
        self.training_progress_updated.connect(self.update_training_progress)
        self.training_metrics_updated.connect(self.update_training_metrics)
        
        logger.info("CustomTrainingApp UI initialized.")

    @pyqtSlot(int, int, str)
    def update_training_progress(self, current_epoch, total_epochs, status):
        """Update the training progress display"""
        if total_epochs > 0 and current_epoch > 0:
            progress_percentage = int((current_epoch / total_epochs) * 100)
            self.training_progress_bar.setValue(progress_percentage)
            self.training_progress_bar.setFormat(f"{progress_percentage}% Complete")
            
            # Update epoch counter
            self.epoch_counter_label.setText(f"Epoch {current_epoch} / {total_epochs}")
            
            # Show progress components
            self.training_progress_bar.setVisible(True)
            self.epoch_counter_label.setVisible(True)
            self.training_time_label.setVisible(True)
            self.metrics_label.setVisible(True)
        
        # Update status
        self.trainingStatusLabel.setText(f"Status: {status}")
        
        # Handle training completion
        if current_epoch == 0 and ("completed" in status.lower() or "failed" in status.lower()):
            # Training completed or failed, hide progress components after a moment
            self.training_progress_bar.setVisible(False)
            self.epoch_counter_label.setVisible(False)
            self.training_time_label.setVisible(False)
            self.metrics_label.setVisible(False)

    @pyqtSlot(str)
    def update_training_metrics(self, metrics_text):
        """Update the live training metrics display"""
        # Check if this is time information or actual metrics
        if "Elapsed:" in metrics_text and "ETA:" in metrics_text:
            self.training_time_label.setText(metrics_text)
        else:
            self.metrics_label.setText(metrics_text)
        
    def update_training_instructions(self, mode):
        """Update training instructions based on selected mode"""
        if mode == 'Classification':
            instructions = (
                "Train a YOLOv8 classification model on your annotated images.\n"
                "Requirements:\n"
                "• The data folder must contain subfolders where each subfolder name is a class label\n"
                "• Each class folder must contain example images of that class\n"
                "• For best results, provide at least 50-100 images per class"
            )
        else:  # Detection
            instructions = (
                "Train a YOLOv8 detection model using bounding box annotations.\n"
                "Requirements:\n"
                "• Images with corresponding coordinate files in coordinates/ folder\n"
                "• Coordinate files should contain: label confidence x1 y1 x2 y2\n"
                "• Images can be organized in class subfolders (from unified annotation)\n"
                "• For best results, provide at least 100 bounding boxes per class\n"
                "• Use the Manual Annotator with bounding box mode to create training data"
            )
        self.training_instructions.setText(instructions)

    def openDataFolderDialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder for Annotation/Training")
        if folder:
            self.dataFolder = folder
            self.folder_label.setText(folder)
            # Reset status labels or potentially update annotator if open
            self.trainingStatusLabel.setText('Status: Ready')
            logger.info(f"Data folder selected: {folder}")
        else:
            logger.info("Data folder selection cancelled.")
            
    def selectOutputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Models and Results", self.outputFolder)
        if folder:
            self.outputFolder = folder
            self.outputFolder_label.setText(folder)
            os.makedirs(self.outputFolder, exist_ok=True)
            logger.info(f"Output folder selected: {folder}")
            
    # --- Annotation Tab Methods ---
    def launchAnnotator(self):
        if not self.dataFolder or not os.path.isdir(self.dataFolder):
            Notifications.warning(self, 'Please select a valid data folder first.')
            return

        # Check if annotator is already open for this folder
        if self.annotator_instance and self.annotator_instance.folderPath == self.dataFolder and self.annotator_instance.isVisible():
            self.annotator_instance.activateWindow()
            Notifications.info(self, "Annotator window is already open for this folder.")
            return
        elif self.annotator_instance:
            # Close previous instance if folder changed or it was closed
            self.annotator_instance.close()

        try:
            # Check if folder contains images before launching
            images_in_folder = [img for img in os.listdir(self.dataFolder)
                                if os.path.isfile(os.path.join(self.dataFolder, img)) and img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not images_in_folder:
                Notifications.warning(self, "The selected folder does not contain any supported image files (.jpg, .png, .jpeg) at the root level.")
                return

            self.annotator_instance = ImageAnnotator(self.dataFolder)
            self.annotator_instance.show()
        except Exception as e:
            logger.error(f"Error launching annotator for {self.dataFolder}: {e}", exc_info=True)
            Notifications.error(self, f"Could not launch annotator: {e}")

    def autoAnnotate(self):
        keyword = self.autoAnnotateInput.text().strip().lower()
        if not keyword:
            Notifications.warning(self, 'Please enter a keyword for automatic grouping.')
            return

        if not self.dataFolder or not os.path.isdir(self.dataFolder):
            Notifications.warning(self, 'Please select a data folder first.')
            return

        target_folder = os.path.join(self.dataFolder, keyword)
        os.makedirs(target_folder, exist_ok=True)
        moved_count = 0
        skipped_count = 0

        try:
            for img_filename in os.listdir(self.dataFolder):
                source_path = os.path.join(self.dataFolder, img_filename)
                # Check if it's a file and contains the keyword
                if os.path.isfile(source_path) and keyword in img_filename.lower():
                    destination_path = os.path.join(target_folder, img_filename)
                    try:
                        shutil.move(source_path, destination_path)
                        logger.info(f"Auto-moved {source_path} to {destination_path}")
                        moved_count += 1
                    except Exception as e:
                        logger.error(f"Error auto-moving {source_path} to {destination_path}: {e}")
                        skipped_count += 1
        except Exception as e:
            logger.error(f"Error during auto-annotation directory scan: {e}")
            Notifications.error(self, f"Error scanning folder: {e}")
            return

        if moved_count > 0:
            Notifications.success(self, f'{moved_count} images containing keyword "{keyword}" moved to subfolder. {skipped_count} skipped due to errors.')
        elif skipped_count > 0:
            Notifications.warning(self, f'No images moved. {skipped_count} potential matches skipped due to errors.')
        else:
            Notifications.info(self, f'No images found containing the keyword "{keyword}" at the root of the selected folder.')

    # --- Training Tab Methods ---
    def prepare_classification_dataset(self, data_folder, project_dir, model_name):
        """Prepare classification dataset within the project structure"""
        try:
            # Create datasets folder within the project
            datasets_dir = os.path.join(project_dir, 'datasets')
            os.makedirs(datasets_dir, exist_ok=True)
            
            # Copy the classification data structure to the project datasets folder
            import shutil
            dataset_dest = os.path.join(datasets_dir, 'classification_data')
            
            # Remove existing dataset if it exists
            if os.path.exists(dataset_dest):
                shutil.rmtree(dataset_dest)
                
            # Create the destination directory
            os.makedirs(dataset_dest, exist_ok=True)
            
            # Copy only class folders (exclude coordinates folder and loose files)
            for item in os.listdir(data_folder):
                source_path = os.path.join(data_folder, item)
                
                # Only copy directories that are not 'coordinates'
                if os.path.isdir(source_path) and item != 'coordinates':
                    dest_path = os.path.join(dataset_dest, item)
                    shutil.copytree(source_path, dest_path)
                    logger.info(f"Copied class folder: {item}")
            
            logger.info(f"Classification dataset prepared at {dataset_dest}")
            return os.path.abspath(dataset_dest)
            
        except Exception as e:
            logger.error(f"Error preparing classification dataset: {e}")
            raise

    def prepare_detection_dataset(self, data_folder, output_folder, model_name):
        """Prepare YOLO detection dataset from coordinate files"""
        try:
            # Create YOLO dataset structure within the unified project folder
            project_dir = os.path.join(output_folder, model_name)
            dataset_dir = os.path.join(project_dir, 'datasets')
            images_dir = os.path.join(dataset_dir, 'images', 'train')
            labels_dir = os.path.join(dataset_dir, 'labels', 'train')
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Find coordinate files
            coords_folder = os.path.join(data_folder, 'coordinates')
            if not os.path.exists(coords_folder):
                raise ValueError("No coordinates folder found. Use bounding box annotation mode first.")
                
            coord_files = [f for f in os.listdir(coords_folder) if f.endswith('_coordinates.txt')]
            if not coord_files:
                raise ValueError("No coordinate files found in coordinates folder.")
                
            # Collect all classes
            all_classes = set()
            annotations_data = {}
            
            # Parse coordinate files
            for coord_file in coord_files:
                coord_path = os.path.join(coords_folder, coord_file)
                base_name = coord_file.replace('_coordinates.txt', '')
                
                # Find corresponding image (check root folder and subfolders)
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_path = None
                
                # First check root folder
                for ext in image_extensions:
                    potential_path = os.path.join(data_folder, base_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                # If not found in root, check subfolders (for images organized by class)
                if not image_path:
                    for item in os.listdir(data_folder):
                        subfolder_path = os.path.join(data_folder, item)
                        if os.path.isdir(subfolder_path) and item != 'coordinates':
                            for ext in image_extensions:
                                potential_path = os.path.join(subfolder_path, base_name + ext)
                                if os.path.exists(potential_path):
                                    image_path = potential_path
                                    break
                            if image_path:
                                break
                        
                if not image_path:
                    logger.warning(f"No image found for {coord_file}")
                    continue
                    
                # Parse coordinate file
                with open(coord_path, 'r') as f:
                    lines = f.readlines()
                    
                boxes = []
                for line in lines[1:]:  # Skip header line
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        label = parts[0]
                        x1, y1, x2, y2 = map(int, parts[2:6])
                        boxes.append({'label': label, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                        all_classes.add(label)
                        
                if boxes:
                    annotations_data[image_path] = boxes
                    
            if not annotations_data:
                raise ValueError("No valid annotations found.")
                
            # Create class mapping
            class_names = sorted(list(all_classes))
            class_to_id = {name: i for i, name in enumerate(class_names)}
            
            # Copy images and create YOLO labels
            for image_path, boxes in annotations_data.items():
                # Copy image
                image_name = os.path.basename(image_path)
                dest_image_path = os.path.join(images_dir, image_name)
                shutil.copy2(image_path, dest_image_path)
                
                # Get image dimensions
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    
                # Create YOLO label file
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                with open(label_path, 'w') as f:
                    for box in boxes:
                        # Convert to YOLO format (normalized center_x, center_y, width, height)
                        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                        center_x = (x1 + x2) / 2.0 / img_width
                        center_y = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        class_id = class_to_id[box['label']]
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        
            # Create dataset.yaml
            yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
            with open(yaml_path, 'w') as f:
                f.write(f"path: {dataset_dir}\n")
                f.write("train: images/train\n")
                f.write("val: images/train\n")  # Use same data for validation for now
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
                
            logger.info(f"Detection dataset prepared: {len(annotations_data)} images, {len(class_names)} classes")
            return yaml_path, len(annotations_data)
            
        except Exception as e:
            logger.error(f"Error preparing detection dataset: {e}")
            raise
    
    def runTraining(self):
        training_mode = self.training_mode_combo.currentText()
        model_size = self.modelSizeComboBox.currentText()
        epochs_str = self.epochsLineEdit.text().strip()
        model_name = self.project_name_input.text().strip()
        img_size = self.imgSizeComboBox.currentText()
        batch_size_str = self.batchSizeLineEdit.text().strip()
        device = self.deviceComboBox.currentText()
        
        # Get advanced parameters if the section is enabled
        advanced_enabled = self.findChild(QGroupBox, "Advanced Options").isChecked()
        lr = self.learningRateLineEdit.text().strip() if advanced_enabled else None
        patience = self.patienceLineEdit.text().strip() if advanced_enabled else None
        augment = self.augCheckBox.isChecked() if advanced_enabled else True
        optimizer = self.optimizerComboBox.currentText() if advanced_enabled else "AdamW"
        save_best = self.saveBestCheckBox.isChecked() if advanced_enabled else True

        if not self.dataFolder or not os.path.isdir(self.dataFolder):
            Notifications.warning(self, 'Please select a training data folder.')
            return

        if not epochs_str.isdigit() or int(epochs_str) <= 0:
            Notifications.warning(self, 'Please enter a valid positive number of epochs.')
            return
        epochs = int(epochs_str)
        
        if not batch_size_str.isdigit() or int(batch_size_str) <= 0:
            Notifications.warning(self, 'Please enter a valid positive batch size.')
            return
        batch_size = int(batch_size_str)

        if not model_name:
            Notifications.warning(self, 'Please enter a project name (used for the output folder).')
            return

        # Validation based on training mode
        if training_mode == 'Classification':
            # Check if the data folder has class subdirectories (excluding coordinates folder)
            class_folders = [item for item in os.listdir(self.dataFolder) 
                           if os.path.isdir(os.path.join(self.dataFolder, item)) and item != 'coordinates']
            
            if not class_folders:
                Notifications.warning(self, 'For classification training, the data folder must contain class subdirectories (excluding the coordinates folder). Each subdirectory name should be a class label containing the corresponding images.')
                return
            else:
                logger.info(f"Found {len(class_folders)} class folders for classification: {class_folders}")
        else:  # Detection
            # Check if coordinates folder exists
            coords_folder = os.path.join(self.dataFolder, 'coordinates')
            if not os.path.exists(coords_folder):
                Notifications.warning(self, 'For detection training, please use the Manual Annotator with bounding box mode to create coordinate files first.')
                return

        # Create unified project directory structure
        project_dir = os.path.join(self.outputFolder, model_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Set up task-specific subdirectories and names
        if training_mode == 'Classification':
            task_name = "classify"
            model_suffix = "-cls"
        else:  # Detection
            task_name = "detect"
            model_suffix = ""
        
        yolo_executable = ".venv/bin/yolo" # Assuming virtual env usage as per instructions
        if not os.path.exists(yolo_executable):
            # Try finding yolo in PATH if not in venv (less ideal)
            yolo_executable_path = shutil.which("yolo")
            if not yolo_executable_path:
                Notifications.error(self, "Could not find the 'yolo' command. Make sure Ultralytics is installed and accessible in your environment (ideally run 'uv pip install ultralytics').")
                return
            else:
                yolo_executable = yolo_executable_path
                logger.warning(f"Using YOLO found in PATH: {yolo_executable}")

        # Create models directory for YOLO architectures
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo_architectures')
        models_dir = os.path.abspath(models_dir)  # Get absolute path
        os.makedirs(models_dir, exist_ok=True)
        
        # Base model filename and path
        model_filename = f'yolov8{model_size}{model_suffix}.pt'
        model_path = os.path.join(models_dir, model_filename)
        
        # Check if model exists in current directory (legacy location) and move it
        current_dir_model = os.path.join(os.getcwd(), model_filename)
        if os.path.exists(current_dir_model) and not os.path.exists(model_path):
            try:
                import shutil
                shutil.move(current_dir_model, model_path)
                logger.info(f"Moved existing model from {current_dir_model} to {model_path}")
            except Exception as e:
                logger.warning(f"Could not move existing model: {e}")
        
        # Pre-download model if it doesn't exist
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_filename} to {models_dir}...")
            try:
                # Download directly to our target location using YOLO's download function
                from ultralytics.utils.downloads import download
                from ultralytics.utils import ASSETS_URL
                
                download_url = f"{ASSETS_URL}/{model_filename}"
                download(download_url, dir=models_dir, unzip=False)
                
                # Verify the file was downloaded
                if os.path.exists(model_path):
                    logger.info(f"Model {model_filename} successfully downloaded to {model_path}")
                else:
                    raise FileNotFoundError(f"Download completed but file not found at {model_path}")
                    
            except Exception as e:
                logger.warning(f"Could not download model to custom location: {e}")
                try:
                    # Fallback: Initialize YOLO model and find where it downloads
                    from ultralytics import YOLO
                    temp_model = YOLO(model_filename)  # This downloads to ultralytics cache
                    
                    # Try to find the cached model and copy it
                    import shutil
                    from ultralytics.utils import USER_CONFIG_DIR
                    
                    # Common cache locations
                    cache_locations = [
                        os.path.join(USER_CONFIG_DIR, 'weights', model_filename),
                        os.path.join(os.path.expanduser('~'), '.ultralytics', 'weights', model_filename),
                        model_filename  # Current directory fallback
                    ]
                    
                    source_path = None
                    for cache_path in cache_locations:
                        if os.path.exists(cache_path):
                            source_path = cache_path
                            break
                    
                    if source_path and source_path != model_filename:
                        shutil.copy2(source_path, model_path)
                        logger.info(f"Copied model from cache {source_path} to {model_path}")
                    else:
                        logger.warning(f"Using fallback model path: {model_filename}")
                        model_path = model_filename
                        
                except Exception as e2:
                    logger.warning(f"Fallback method also failed: {e2}. Using model filename only.")
                    model_path = model_filename
        
        # Prepare data path based on training mode
        if training_mode == 'Classification':
            try:
                abs_data_path = self.prepare_classification_dataset(self.dataFolder, project_dir, model_name)
                logger.info(f"Classification dataset prepared at {abs_data_path}")
            except Exception as e:
                Notifications.error(self, f"Error preparing classification dataset: {str(e)}")
                return
        else:  # Detection
            try:
                dataset_yaml, num_images = self.prepare_detection_dataset(self.dataFolder, self.outputFolder, model_name)
                abs_data_path = os.path.abspath(dataset_yaml)
                logger.info(f"Detection dataset prepared with {num_images} images")
            except Exception as e:
                Notifications.error(self, f"Error preparing detection dataset: {str(e)}")
                return

        # Create explicit output directory for training results
        training_output_dir = os.path.join(project_dir, task_name)
        
        # Start with basic parameters 
        # Use explicit project/name structure to avoid duplicates
        command = [
            yolo_executable,
            f"task={task_name}",
            "mode=train",
            f"model={model_path}",  # Use full path to models directory
            f"data={abs_data_path}",
            f"epochs={epochs}",
            f"imgsz={img_size}",
            f"batch={batch_size}",
            f"project={project_dir}",  # Main project directory
            f"name={task_name}",  # Task name for the run
            f"device={device}",
            "exist_ok=True"  # Allow overwriting existing runs
        ]
        
        # Add advanced parameters if enabled
        if advanced_enabled:
            if lr:
                try:
                    lr_float = float(lr)
                    command.append(f"lr0={lr_float}")
                except ValueError:
                    Notifications.warning(self, "Invalid learning rate value. Using default.")
            
            if patience and patience.isdigit():
                command.append(f"patience={patience}")
            
            command.append(f"optimizer={optimizer}")
            command.append(f"augment={str(augment).lower()}")
            command.append(f"save_best={str(save_best).lower()}")
        
        # Join the command parts for execution
        command_str = " ".join(command)
        
        logger.info(f"Running training command: {command_str}")
        self.trainingStatusLabel.setText('Status: Training starting...')
        QApplication.processEvents()

        # Run in a separate thread to avoid blocking the GUI
        def train_thread():
            import time
            start_time = time.time()
            
            try:
                # Run the command, capture output with real-time streaming
                process = subprocess.Popen(command_str, shell=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=0, universal_newlines=True,
                                           env=dict(os.environ, PYTHONUNBUFFERED='1'))

                # Progress tracking variables
                current_epoch = 0
                total_epochs = epochs
                latest_metrics = {}
                
                # Emit initial progress
                self.training_progress_updated.emit(0, total_epochs, "Training starting...")
                
                # Add a small counter to track lines processed
                line_count = 0
                last_progress_update = time.time()

                # Log output line by line and parse for progress
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if not line:  # EOF
                            break
                            
                        line_clean = line.strip()
                        line_count += 1
                        
                        if line_clean:  # Only log non-empty lines
                            logger.info(f"[YOLO Train #{line_count}]: {line_clean}")
                            
                            # Debug: Check for potential epoch lines
                            if '/' in line_clean and any(c.isdigit() for c in line_clean):
                                import re
                                if re.search(r'\d+/\d+', line_clean):
                                    logger.info(f"🔎 Potential epoch line detected: '{line_clean[:100]}'")
                            
                            # Force UI update every few lines to keep responsive
                            if line_count % 10 == 0:
                                from PyQt5.QtWidgets import QApplication
                                QApplication.processEvents()
                        
                        # Parse YOLO training output for progress information with multiple patterns
                        # Look for lines that might contain epoch information
                        if line_clean and ('/' in line_clean and any(char.isdigit() for char in line_clean)):
                            try:
                                import re
                                
                                # YOLO v8 specific patterns based on actual output
                                patterns = [
                                    r'^(\d+)/(\d+)\s+[\d.]+G\s+',  # "1/10      23.2G      " - exact YOLO format
                                    r'(\d+)/(\d+)\s+[\d.]+G',      # "1/10      23.2G" - main YOLO format
                                    r'^(\d+)/(\d+)\s+',            # "1/10      " at start of line
                                    r'(\d+)/(\d+)\s*:.*640',       # "1/10:" followed by resolution
                                    r'epoch[:\s]*(\d+)[/:](\d+)',  # "Epoch 1/100" or "epoch: 1/100"
                                    r'(\d+)/(\d+)\s*epochs',       # "1/100 epochs"
                                ]
                                
                                epoch_found = False
                                for i, pattern in enumerate(patterns):
                                    epoch_match = re.search(pattern, line_clean, re.IGNORECASE)
                                    if epoch_match:
                                        try:
                                            current_epoch = int(epoch_match.group(1))
                                            detected_total = int(epoch_match.group(2))
                                            logger.info(f"🔍 Pattern {i+1} matched: '{pattern}' in line: '{line_clean[:100]}' -> Epoch {current_epoch}/{detected_total}")
                                            
                                            # Use detected total if it makes sense, otherwise use expected total
                                            if detected_total > 0 and detected_total <= 10000:  # Sanity check
                                                total_epochs = detected_total
                                            
                                            # Calculate elapsed time and ETA
                                            elapsed_time = time.time() - start_time
                                            if current_epoch > 0:
                                                avg_time_per_epoch = elapsed_time / current_epoch
                                                eta_seconds = avg_time_per_epoch * (total_epochs - current_epoch)
                                                eta_str = f"ETA: {int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
                                            else:
                                                eta_str = "ETA: Calculating..."
                                            
                                            # Update time display
                                            elapsed_str = f"Elapsed: {int(elapsed_time//3600):02d}:{int((elapsed_time%3600)//60):02d}:{int(elapsed_time%60):02d}"
                                            self.training_metrics_updated.emit(f"{elapsed_str} | {eta_str}")
                                            
                                            # Emit progress update
                                            self.training_progress_updated.emit(current_epoch, total_epochs, f"Training epoch {current_epoch}/{total_epochs}")
                                            logger.info(f"🎯 Progress updated: Epoch {current_epoch}/{total_epochs} ({(current_epoch/total_epochs)*100:.1f}%)")
                                            last_progress_update = time.time()
                                            epoch_found = True
                                            break
                                            
                                        except (ValueError, IndexError):
                                            continue
                                
                                # If no epoch found, try to extract from generic progress indicators
                                if not epoch_found:
                                    # Look for percentage indicators
                                    percent_match = re.search(r'(\d+)%', line_clean)
                                    if percent_match and 'epoch' in line_clean.lower():
                                        try:
                                            progress_percent = int(percent_match.group(1))
                                            estimated_epoch = int((progress_percent / 100.0) * total_epochs)
                                            if estimated_epoch > current_epoch:
                                                current_epoch = estimated_epoch
                                                self.training_progress_updated.emit(current_epoch, total_epochs, f"Training progress {progress_percent}% (epoch ~{current_epoch})")
                                        except (ValueError, IndexError):
                                            pass
                                    
                            except Exception as e:
                                logger.debug(f"Error parsing epoch from line '{line_clean}': {e}")
                                pass
                        
                        # Parse metrics (loss, accuracy, etc.)
                        elif any(metric in line_clean.lower() for metric in ['loss', 'precision', 'recall', 'map50', 'accuracy']):
                            try:
                                # Extract key metrics from the line
                                if "loss" in line_clean.lower():
                                    # Look for loss values
                                    import re
                                    loss_matches = re.findall(r'loss:?\s*([0-9.]+)', line_clean, re.IGNORECASE)
                                    if loss_matches:
                                        latest_metrics['loss'] = float(loss_matches[0])
                                
                                if "precision" in line_clean.lower():
                                    precision_matches = re.findall(r'precision:?\s*([0-9.]+)', line_clean, re.IGNORECASE)
                                    if precision_matches:
                                        latest_metrics['precision'] = float(precision_matches[0])
                                
                                if "recall" in line_clean.lower():
                                    recall_matches = re.findall(r'recall:?\s*([0-9.]+)', line_clean, re.IGNORECASE)
                                    if recall_matches:
                                        latest_metrics['recall'] = float(recall_matches[0])
                                
                                if "map50" in line_clean.lower():
                                    map_matches = re.findall(r'map50:?\s*([0-9.]+)', line_clean, re.IGNORECASE)
                                    if map_matches:
                                        latest_metrics['mAP@50'] = float(map_matches[0])
                                
                                if "accuracy" in line_clean.lower():
                                    acc_matches = re.findall(r'accuracy:?\s*([0-9.]+)', line_clean, re.IGNORECASE)
                                    if acc_matches:
                                        latest_metrics['accuracy'] = float(acc_matches[0])
                                
                                # Update metrics display
                                if latest_metrics:
                                    metrics_text = " | ".join([f"{k}: {v:.4f}" for k, v in latest_metrics.items()])
                                    self.training_metrics_updated.emit(f"Epoch {current_epoch}: {metrics_text}")
                                    
                            except (ValueError, IndexError, AttributeError):
                                pass
                        
                        # Force periodic progress updates if no epoch info detected for a while
                        current_time = time.time()
                        if current_time - last_progress_update > 30:  # No progress update for 30 seconds
                            # Estimate progress based on time elapsed
                            elapsed_time = current_time - start_time
                            if elapsed_time > 60:  # Only after 1 minute
                                estimated_progress = min(int((elapsed_time / (total_epochs * 60)) * 100), 95)  # Rough estimate
                                estimated_epoch = min(int((elapsed_time / 60) * 2), total_epochs - 1)  # Very rough estimate
                                if estimated_epoch > current_epoch:
                                    current_epoch = estimated_epoch
                                    self.training_progress_updated.emit(current_epoch, total_epochs, f"Training in progress (estimated epoch {current_epoch})")
                                    logger.info(f"📊 Fallback progress estimate: Epoch ~{current_epoch}/{total_epochs}")
                                    last_progress_update = current_time
                        
                        # Check for training completion or validation messages
                        elif "training complete" in line_clean.lower() or "results saved" in line_clean.lower():
                            self.training_progress_updated.emit(total_epochs, total_epochs, "Training completed successfully!")

                process.wait()
                return_code = process.returncode

                if return_code == 0:
                    logger.info("Training completed successfully.")
                    # Final progress update
                    self.training_progress_updated.emit(0, 0, "Training completed successfully!")
                    # Update status on the main thread
                    QApplication.instance().postEvent(self, TrainingCompleteEvent(True, model_name))
                else:
                    logger.error(f"Training failed with exit code: {return_code}")
                    self.training_progress_updated.emit(0, 0, f"Training failed (exit code {return_code})")
                    QApplication.instance().postEvent(self, TrainingCompleteEvent(False, model_name, f"Training failed (exit code {return_code}). Check logs."))

            except Exception as e:
                logger.error(f"Exception during training thread: {e}", exc_info=True)
                self.training_progress_updated.emit(0, 0, f"Training error: {str(e)}")
                QApplication.instance().postEvent(self, TrainingCompleteEvent(False, model_name, f"Error running training: {e}"))

        threading.Thread(target=train_thread, daemon=True).start()

    # Handle custom event for training completion to update UI safely
    def event(self, event):
        if isinstance(event, TrainingCompleteEvent):
            if event.success:
                self.trainingStatusLabel.setText(f'Status: Training \'{event.model_name}\' completed successfully!')
                Notifications.success(self, f"Training for experiment '{event.model_name}' finished.")
            else:
                self.trainingStatusLabel.setText(f'Status: Training \'{event.model_name}\' failed. {event.message}')
                Notifications.error(self, event.message)
            return True
        return super().event(event)

# Custom Event class for signaling training completion from thread
from PyQt5.QtCore import QEvent
class TrainingCompleteEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 1)
    def __init__(self, success, model_name, message=""):
        super().__init__(self.EVENT_TYPE)
        self.success = success
        self.model_name = model_name
        self.message = message

# Example usage (for testing standalone)
if __name__ == '__main__':
    # Ensure QApplication instance exists for event posting
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = CustomTrainingApp()
    window.show()
    sys.exit(app.exec_()) 