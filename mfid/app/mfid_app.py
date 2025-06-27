import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, 
                           QHBoxLayout, QGridLayout, QFrame, QDesktopWidget, QMenu, 
                           QAction, QMainWindow, QToolBar, QStatusBar)
from PyQt5.QtGui import QColor, QPalette, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QSize
from mfid.utils.config_manager import ConfigManager
from mfid.utils.logging_utils import get_logger, MFIDLogger
from mfid.utils.notifications import Notifications

# Import App Windows (adjust as needed when apps are created/moved)
from mfid.analysis.analysis_app import AnalysisApp
# Updated imports for new apps
from mfid.training.custom_training_app import CustomTrainingApp 
from mfid.settings.settings_app import SettingsApp

# Initialize logger
logger = get_logger('app')

class LauncherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize config and session
        self.config = ConfigManager()
        self.session_log, self.session_handler = MFIDLogger.create_session_log()
        logger.info("MFID application started")
        
        self.init_ui()
        
    def init_ui(self):
        self.apply_dark_theme()
        self.setWindowTitle('MFID - Monkey Face ID')
        self.resize(800, 600)
        self.center_on_screen()
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create header with logo and title
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)
        
        # Create grid of app buttons
        cards_layout = self.create_app_cards()
        main_layout.addLayout(cards_layout)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Show welcome notification
        QApplication.processEvents()
        Notifications.info(self, "Welcome to MFID - Monkey Face ID")
    
    def create_header(self):
        header_layout = QHBoxLayout()
        
        # Load and display the logo
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, '..', '..', 'images', 'main.jpg')
        
        logo_label = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(250, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        
        # Create title label
        title_label = QLabel("MFID - Monkey Face ID")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        # Create subtitle label
        subtitle_label = QLabel("Recognize and identify Rhesus macaques using computer vision")
        subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        
        # Add widgets to header layout
        header_layout.addWidget(logo_label)
        
        # Create a vertical layout for title and subtitle
        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        return header_layout
    
    def create_app_cards(self):
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        
        # Analysis Card - Merged Detection, Classification, and Identity Recognition
        analysis_card = self.create_app_card(
            "Analysis",
            "Comprehensive detection, classification, and identity recognition",
            self.open_analysis_app,
            "icons/analysis.png"
        )
        grid_layout.addWidget(analysis_card, 0, 0)
        
        # Custom Training Card
        training_card = self.create_app_card(
            "Custom Training",
            "Annotate images and train custom models",
            self.open_training_app,
            "icons/training.png"
        )
        grid_layout.addWidget(training_card, 0, 1)
        
        return grid_layout
    
    def create_app_card(self, title, description, on_click, icon_path=None):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2D2D30;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(card)
        
        # Card header with icon and title
        header_layout = QHBoxLayout()
        
        if icon_path and os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', icon_path)):
            icon_label = QLabel()
            pixmap = QPixmap(os.path.join(os.path.dirname(__file__), '..', '..', icon_path))
            pixmap = pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(18)  # Increased from 14 to 18
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        layout.addStretch()
        
        # Launch button
        launch_button = QPushButton(f"Launch {title}")
        launch_button.setCursor(Qt.PointingHandCursor)
        launch_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
            QPushButton:pressed {
                background-color: #005FA3;
            }
        """)
        launch_button.clicked.connect(on_click)
        
        layout.addWidget(launch_button)
        
        return card
    
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
        
        # Set application-wide stylesheet
        self.setStyleSheet("""
            QPushButton { 
                background-color: #4A4A4A; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 4px;
            }
            QPushButton:hover { 
                background-color: #5A5A5A; 
            }
            QLabel, QCheckBox { 
                color: white; 
            }
            QMenuBar {
                background-color: #2D2D30;
                color: white;
            }
            QMenuBar::item {
                background-color: #2D2D30;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #3D3D3D;
            }
            QMenu {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #3D3D3D;
            }
            QStatusBar {
                background-color: #2D2D30;
                color: white;
            }
            QToolBar {
                background-color: #2D2D30;
                border: none;
            }
        """)
    
    def create_menu_bar(self):
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu('File')
        
        # Add actions to file menu
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Applications menu
        apps_menu = menu_bar.addMenu('Applications')
        
        analysis_action = QAction('Analysis', self)
        analysis_action.triggered.connect(self.open_analysis_app)
        apps_menu.addAction(analysis_action)
        
        training_action = QAction('Custom Training', self)
        training_action.triggered.connect(self.open_training_app)
        apps_menu.addAction(training_action)
        
        settings_action_apps = QAction('Settings', self)
        settings_action_apps.triggered.connect(self.open_settings_app)
        apps_menu.addAction(settings_action_apps)
        
        # Add Settings as a direct menu item in the top bar
        settings_action = QAction('⚙ Settings', self)
        settings_action.setStatusTip('Open application settings')
        settings_action.triggered.connect(self.open_settings_app)
        menu_bar.addAction(settings_action)
        
        # Help menu
        help_menu = menu_bar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_action = QAction('Documentation', self)
        help_action.triggered.connect(self.show_documentation)
        help_menu.addAction(help_action)
    
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24,24)) # Consistent icon size
        self.addToolBar(toolbar)

        # Get base path for icons
        base_icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'icons')

        # Analysis Action
        analysis_action = QAction(QIcon(os.path.join(base_icon_path, 'analysis.png')) if os.path.exists(os.path.join(base_icon_path, 'analysis.png')) else QIcon(), "&Analysis", self)
        analysis_action.setStatusTip("Open Analysis App (Detection, Classification & Identity)")
        analysis_action.triggered.connect(self.open_analysis_app)
        toolbar.addAction(analysis_action)
        
        # Add Custom Training action
        training_action = QAction(QIcon('icons/training.png') if os.path.exists('icons/training.png') else QIcon(), 'Custom Training', self)
        training_action.triggered.connect(self.open_training_app)
        toolbar.addAction(training_action)
        
        # Add Settings action
        settings_action = QAction(QIcon('icons/settings.png') if os.path.exists('icons/settings.png') else QIcon(), '⚙ Settings', self)
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self.open_settings_app)
        toolbar.addAction(settings_action)
    
    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def open_analysis_app(self):
        logger.info("Opening Analysis App")
        self.status_bar.showMessage('Opening Analysis App...')
        try:
            # Ensure any previous instance is properly closed if desired, or manage instances
            if not hasattr(self, 'analysis_app_instance') or not self.analysis_app_instance.isVisible():
                self.analysis_app_instance = AnalysisApp()
                self.analysis_app_instance.show()
            else:
                self.analysis_app_instance.activateWindow()
            self.status_bar.showMessage('Analysis App opened')
        except Exception as e:
            logger.error(f"Error opening Analysis App: {e}", exc_info=True)
            Notifications.error(self, f"Could not open Analysis App: {e}")
            self.status_bar.showMessage('Error opening Analysis App')
    
    def open_settings(self):
        logger.info("Opening Settings (redirecting to new Settings App)")
        self.open_settings_app()  # Redirect to the new settings app for backward compatibility
    
    def show_about(self):
        logger.info("Showing About dialog")
        Notifications.info(self, "MFID - Monkey Face ID\nA tool for recognizing Rhesus macaques in videos and images")
    
    def show_documentation(self):
        logger.info("Showing Documentation")
        Notifications.info(self, "Documentation will be available in a future update")
    
    def closeEvent(self, event):
        """Handle application close event"""
        logger.info("MFID application closing")
        
        # Clean up resources
        MFIDLogger.end_session_log(self.session_handler)
        
        # Accept the close event
        event.accept()

    # Updated method for training app
    def open_training_app(self):
        logger.info("Opening Custom Training App")
        self.status_bar.showMessage('Opening Custom Training App...')
        try:
            # Check if window already exists and show it, otherwise create
            if not hasattr(self, 'training_app_window') or not self.training_app_window.isVisible():
                self.training_app_window = CustomTrainingApp()
                self.training_app_window.show()
            else:
                self.training_app_window.activateWindow() # Bring to front if already open
                
            self.status_bar.showMessage('Custom Training App opened')
        except Exception as e:
            logger.error(f"Error opening Custom Training App: {e}", exc_info=True)
            Notifications.error(self, f"Error opening Custom Training App: {str(e)}")
            self.status_bar.showMessage('Error opening Custom Training App')

    # Updated method for settings app
    def open_settings_app(self):
        logger.info("Opening Settings App")
        self.status_bar.showMessage('Opening Settings App...')
        try:
             # Check if window already exists and show it, otherwise create
            if not hasattr(self, 'settings_app_window') or not self.settings_app_window.isVisible():
                self.settings_app_window = SettingsApp()
                self.settings_app_window.show()
            else:
                self.settings_app_window.activateWindow() # Bring to front if already open
                
            self.status_bar.showMessage('Settings App opened')
        except Exception as e:
            logger.error(f"Error opening Settings App: {e}", exc_info=True)
            Notifications.error(self, f"Error opening Settings App: {str(e)}")
            self.status_bar.showMessage('Error opening Settings App')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LauncherApp()
    ex.show()
    sys.exit(app.exec_())
