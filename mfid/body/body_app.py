import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QFormLayout, QMessageBox, QDoubleSpinBox, QComboBox, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QFont
from ultralytics import YOLO
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit

class BodyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MonkeySorter")
        self.resize(700, 300)
        self.init_ui()

    def init_ui(self):
        # Modern UI Design Adjustments
        DarkTheme(self) 
        layout = QVBoxLayout()
        params_layout = self.create_parameters_layout()
        layout.addLayout(params_layout)
        
        # Buttons with modern styling
        select_video_folder_button = DarkButton("Select Videos Folder", self.select_video_folder)
        layout.addWidget(select_video_folder_button)

        run_sorting_button = DarkButton("Run Sorting", self.run_sorting)
        layout.addWidget(run_sorting_button)

        self.setLayout(layout)


    def create_parameters_layout(self):
        params_layout = QFormLayout()
        self.model_box, self.conf_box, self.iou_box, self.device_box = QComboBox(), QDoubleSpinBox(), QDoubleSpinBox(), QComboBox()
        self.show_box, self.save_box, self.save_txt_box = QCheckBox(), QCheckBox(), QCheckBox()

        for box in [self.model_box, self.device_box]:
            box.setStyleSheet("background-color: #323232; color: white;")

        for spin_box in [self.conf_box, self.iou_box]:
            spin_box.setRange(0, 1)
            spin_box.setSingleStep(0.01)
            spin_box.setValue(0.5)
            spin_box.setStyleSheet("background-color: #323232; color: white;")

        self.iou_box.setValue(0.7)
        self.setup_model_box()
        self.setup_device_box()

        for item, label in [(self.model_box, "Model:"), (self.conf_box, "Confidence:"), (self.iou_box, "IOU:"),
                            (self.device_box, "Device:"), (self.show_box, "Show:"), (self.save_box, "Save:"),
                            (self.save_txt_box, "Save TXT:")]:
            params_layout.addRow(label, item)

        return params_layout

    def setup_model_box(self):
        for model_size in ["n", "s", "m", "l"]:
            self.model_box.addItem(model_size)

    def setup_device_box(self):
        for device in ["None", "cpu", "0"]:
            self.device_box.addItem(device)

    def create_button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setStyleSheet("QPushButton { background-color: #4A4A4A; color: white; border: none; padding: 5px; }"
                             "QPushButton:hover { background-color: #5A5A5A; }")
        return button


    def select_video_folder(self):
        self.video_folder = QFileDialog.getExistingDirectory()

    def process_videos(self, video_folder):

        params = {
            "model": self.model_box.currentText(),
            "conf": self.conf_box.value(),
            "iou": self.iou_box.value(),
            "device": self.device_box.currentText(),
            "show": self.show_box.isChecked(),
            "save": self.save_box.isChecked(),
            "save_txt": self.save_txt_box.isChecked()
        }

        if params["device"] == "None":
            params["device"] = None

        script_directory = os.path.dirname(__file__)  
        model_directory = os.path.normpath(os.path.join(script_directory, '..', 'models'))
        model_name = f"best_{params['model']}.pt"
        self.model = YOLO(os.path.join(model_directory, model_name))

        detection_folder = os.path.join(video_folder, "detections")
        no_detection_folder = os.path.join(video_folder, "no_detections")

        os.makedirs(detection_folder, exist_ok=True)
        os.makedirs(no_detection_folder, exist_ok=True)

        for file in os.listdir(video_folder):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(video_folder, file)
                results = self.model.predict(video_path, **params)

                has_detections = False
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if len(box.xyxy[0]) > 0:
                            has_detections = True
                            break

                output_folder = detection_folder if has_detections else no_detection_folder
                output_path = os.path.join(output_folder, file)

                shutil.copy2(video_path, output_path)
                os.remove(video_path)

    def run_sorting(self):
        if self.video_folder is None:
            QMessageBox.warning(self, "Error", "Please selelct videos directory.")
            return

        self.process_videos(self.video_folder)
        QMessageBox.about(self, "Sorting completed", "Videos folder has been successfully sorted.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BodyApp()
    window.show()
    sys.exit(app.exec_())
