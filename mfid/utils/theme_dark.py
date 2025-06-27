from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

def DarkTheme(widget):
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
    widget.setPalette(dark_palette)
    widget.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                         "QPushButton:hover { background-color: #5A5A5A; }"
                         "QLabel, QCheckBox { color: white; }")

def DarkButton(text, function):
    button = QPushButton(text)
    button.clicked.connect(function)
    button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                         "QPushButton:hover { background-color: #5A5A5A; }")
    return button

def DarkLineEdit(line_edit_widget, placeholder_text):
    line_edit_widget.setPlaceholderText(placeholder_text)
    # Set light grey background with black text for maximum visibility
    line_edit_widget.setStyleSheet("""
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
        QLineEdit:focus {
            border: 2px solid #2196F3;
            background-color: #F0F0F0;
        }
    """)
