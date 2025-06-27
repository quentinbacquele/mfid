import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFormLayout, QLineEdit, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from mfid.utils.theme_dark import DarkTheme, DarkButton
from mfid.utils.logging_utils import get_logger
from mfid.utils.config_manager import ConfigManager
from mfid.utils.notifications import Notifications

logger = get_logger('settings_app')

class SettingsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        # Set default output folder to mfid directory
        self.default_output_folder = os.path.join(os.path.dirname(__file__), '..', '..')
        self.default_output_folder = os.path.abspath(self.default_output_folder)  # Get absolute path
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Application Settings')
        self.resize(600, 200) # Adjusted size
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # --- Default Output Directory Setting --- 
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setStyleSheet("""
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
        self.output_dir_edit.setPlaceholderText(f"Default: {self.default_output_folder}")
        browse_output_button = DarkButton("Browse...", self.select_output_dir)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_output_button)
        form_layout.addRow("Default Output Directory:", output_dir_layout)
        # --- Add more settings controls here as needed --- 
        # Example: 
        # self.some_setting_edit = QLineEdit()
        # form_layout.addRow("Some Other Setting:", self.some_setting_edit)
        
        layout.addLayout(form_layout)
        layout.addStretch() # Push save button to bottom
        
        save_button = DarkButton("Save Settings", self.save_settings) # Use DarkButton with function parameter
        layout.addWidget(save_button, alignment=Qt.AlignRight)
        
        self.load_settings() # Load existing settings on init
        logger.info("SettingsApp UI initialized.")
        
    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Default Output Directory", self.output_dir_edit.text()) # Start browsing from current value
        if folder:
            self.output_dir_edit.setText(folder)
            
    def load_settings(self):
        logger.info("Loading settings...")
        try:
            # Load default output directory
            default_output = self.config.get('paths', 'default_output_directory', self.default_output_folder) # Use mfid directory as fallback default
            self.output_dir_edit.setText(default_output)
            # --- Load other settings here --- 
            # Example:
            # some_value = self.config.get('section', 'setting', 'default_value')
            # self.some_setting_edit.setText(some_value)
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
            Notifications.error(self, "Load Settings Error", f"Could not load settings: {e}")
        
    def save_settings(self):
        logger.info("Saving settings...")
        try:
            # Save default output directory
            output_dir = self.output_dir_edit.text().strip()
            if not output_dir:
                # Maybe enforce a default or use the fallback? For now, save empty if cleared.
                logger.warning("Default output directory cleared by user.")
            self.config.set('paths', 'default_output_directory', output_dir)
            # --- Save other settings here --- 
            # Example:
            # self.config.set('section', 'setting', self.some_setting_edit.text())
            
            self.config.save_config()
            logger.info("Settings saved successfully.")
            Notifications.success(self, "Settings Saved", "Application settings have been updated.")
            self.close() # Close after saving
        except Exception as e:
            logger.error(f"Error saving settings: {e}", exc_info=True)
            Notifications.error(self, "Save Settings Error", f"Could not save settings: {e}")

# Example usage (for testing standalone)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SettingsApp()
    window.show()
    sys.exit(app.exec_()) 