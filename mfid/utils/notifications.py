from PyQt5.QtWidgets import QMessageBox, QLabel, QProgressBar, QVBoxLayout, QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QPoint
from PyQt5.QtGui import QColor, QPalette, QFont
import time

class Toast(QWidget):
    """
    A toast notification widget that shows a message briefly and then fades away
    
    Example:
    ```
    Toast.show_toast(parent_widget, "File saved successfully", Toast.SUCCESS)
    ```
    """
    
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    
    def __init__(self, parent, message, toast_type=INFO, duration=3000):
        """
        Initialize a toast notification
        
        Args:
            parent (QWidget): Parent widget
            message (str): Message to display
            toast_type (str): Type of notification (info, success, warning, error)
            duration (int): Duration in milliseconds
        """
        super().__init__(parent)
        
        # Set up the toast appearance
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Set the background color based on the notification type
        self.setAutoFillBackground(True)
        palette = self.palette()
        
        if toast_type == self.SUCCESS:
            palette.setColor(QPalette.Window, QColor(46, 125, 50, 230))  # Dark green
        elif toast_type == self.WARNING:
            palette.setColor(QPalette.Window, QColor(237, 108, 2, 230))  # Dark orange
        elif toast_type == self.ERROR:
            palette.setColor(QPalette.Window, QColor(198, 40, 40, 230))  # Dark red
        else:  # INFO
            palette.setColor(QPalette.Window, QColor(25, 118, 210, 230))  # Dark blue
        
        self.setPalette(palette)
        
        # Create and style the message label
        self.label = QLabel(message)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 14px; padding: 20px;")
        font = QFont()
        font.setBold(True)
        self.label.setFont(font)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        # Size and position
        self.duration = duration
        self.setFixedWidth(300)
        self.adjustSize()
        
        # Center the toast at the bottom of the parent
        parentRect = parent.geometry()
        geo = self.geometry()
        geo.moveCenter(parentRect.center())
        geo.moveBottom(parentRect.bottom() - 20)
        self.setGeometry(geo)
        
        # Set up animation
        self.fadeIn()
        
        # Set up timer to trigger fade out
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.fadeOut)
        self.timer.start(duration)
    
    def fadeIn(self):
        """Animate the toast fading in"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()
        self.show()
    
    def fadeOut(self):
        """Animate the toast fading out"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(self.close)
        self.animation.start()
    
    @classmethod
    def show_toast(cls, parent, message, toast_type=INFO, duration=3000):
        """
        Show a toast notification
        
        Args:
            parent (QWidget): Parent widget
            message (str): Message to display
            toast_type (str): Type of notification (info, success, warning, error)
            duration (int): Duration in milliseconds
            
        Returns:
            Toast: The toast instance
        """
        toast = cls(parent, message, toast_type, duration)
        return toast


class ProgressNotification(QWidget):
    """
    A progress notification widget that shows a progress bar with a message
    
    Example:
    ```
    progress = ProgressNotification.show_progress(parent_widget, "Processing files...")
    # Update progress
    progress.update_progress(50, "Processing file 5 of 10")
    # Complete
    progress.complete("Processing completed successfully")
    ```
    """
    
    def __init__(self, parent, message):
        """
        Initialize a progress notification
        
        Args:
            parent (QWidget): Parent widget
            message (str): Message to display
        """
        super().__init__(parent)
        
        # Set up the notification appearance
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set the background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(50, 50, 50, 230))
        self.setPalette(palette)
        
        # Create and style the message label
        self.label = QLabel(message)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 14px; padding: 10px;")
        
        # Create and style the progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid white;
                background-color: #3d3d3d;
                color: white;
                text-align: center;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
                margin: 0px;
                border-radius: 3px;
            }
        """)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        
        # Size and position
        self.setMinimumWidth(400)
        self.adjustSize()
        
        # Center the notification at the bottom of the parent
        parentRect = parent.geometry()
        geo = self.geometry()
        geo.moveCenter(parentRect.center())
        self.setGeometry(geo)
        
        # Set up animation
        self.fadeIn()
    
    def fadeIn(self):
        """Animate the notification fading in"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()
        self.show()
    
    def fadeOut(self):
        """Animate the notification fading out"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(self.close)
        self.animation.start()
    
    def update_progress(self, value, message=None):
        """
        Update the progress bar value and optionally the message
        
        Args:
            value (int): Progress value (0-100)
            message (str, optional): New message to display
        """
        self.progress_bar.setValue(value)
        
        if message:
            self.label.setText(message)
        
        # Ensure the UI updates
        QApplication.processEvents()
    
    def complete(self, message=None):
        """
        Complete the progress and show a success message
        
        Args:
            message (str, optional): Completion message to display
        """
        self.progress_bar.setValue(100)
        
        if message:
            self.label.setText(message)
        
        # Ensure the UI updates
        QApplication.processEvents()
        
        # Delay before fade out
        QTimer.singleShot(1500, self.fadeOut)
    
    def error(self, message):
        """
        Show an error message and change the progress bar to red
        
        Args:
            message (str): Error message to display
        """
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid white;
                background-color: #3d3d3d;
                color: white;
                text-align: center;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #c62828;
                width: 10px;
                margin: 0px;
                border-radius: 3px;
            }
        """)
        
        self.label.setText(message)
        
        # Ensure the UI updates
        QApplication.processEvents()
        
        # Delay before fade out
        QTimer.singleShot(3000, self.fadeOut)
    
    @classmethod
    def show_progress(cls, parent, message):
        """
        Show a progress notification
        
        Args:
            parent (QWidget): Parent widget
            message (str): Message to display
            
        Returns:
            ProgressNotification: The progress notification instance
        """
        progress = cls(parent, message)
        return progress


class Notifications:
    """
    Utility class for showing different types of notifications
    
    Example:
    ```
    # Show a simple notification
    Notifications.info(parent_widget, "Information message")
    
    # Show a confirmation dialog
    if Notifications.confirm(parent_widget, "Are you sure?"):
        # User confirmed
        pass
    
    # Show a progress notification
    progress = Notifications.progress(parent_widget, "Processing files...")
    # Update progress
    progress.update_progress(50, "Processing file 5 of 10")
    # Complete
    progress.complete("Processing completed successfully")
    ```
    """
    
    @staticmethod
    def info(parent, message, duration=3000):
        """Show an information notification"""
        return Toast.show_toast(parent, message, Toast.INFO, duration)
    
    @staticmethod
    def success(parent, message, duration=3000):
        """Show a success notification"""
        return Toast.show_toast(parent, message, Toast.SUCCESS, duration)
    
    @staticmethod
    def warning(parent, message, duration=3000):
        """Show a warning notification"""
        return Toast.show_toast(parent, message, Toast.WARNING, duration)
    
    @staticmethod
    def error(parent, message, duration=3000):
        """Show an error notification"""
        return Toast.show_toast(parent, message, Toast.ERROR, duration)
    
    @staticmethod
    def confirm(parent, message, title="Confirm"):
        """Show a confirmation dialog and return True if confirmed"""
        reply = QMessageBox.question(
            parent, title, message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    @staticmethod
    def progress(parent, message):
        """Show a progress notification"""
        return ProgressNotification.show_progress(parent, message) 