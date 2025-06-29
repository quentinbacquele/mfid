import threading
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox, QLineEdit
from mfid.utils.theme_dark import DarkTheme, DarkButton

class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Training Settings')
        self.resize(400, 300)

        layout = QVBoxLayout()

        self.loadFolderButton = DarkButton('Load Folder for training', self.openFolderDialog)
        self.modelSizeComboBox = QComboBox(self)
        self.modelSizeComboBox.addItems(['n', 's', 'm', 'l'])
        self.epochsLineEdit = QLineEdit(self)
        self.epochsLineEdit.setPlaceholderText('Enter number of epochs (e.g., 150)')
        self.modelNameLineEdit = QLineEdit(self)  # Add QLineEdit for model name
        self.modelNameLineEdit.setPlaceholderText('Enter model name')
        self.trainingButton = DarkButton('Train your model', self.runTraining)
        self.statusLabel = QLabel('Status: Ready to load training data', self)

        layout.addWidget(self.loadFolderButton)
        layout.addWidget(self.modelSizeComboBox)
        layout.addWidget(self.epochsLineEdit)
        layout.addWidget(self.modelNameLineEdit)  # Add the model name QLineEdit to the layout
        layout.addWidget(self.trainingButton)
        layout.addWidget(self.statusLabel)

        self.setLayout(layout)

    def openFolderDialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for Training")
        if folder:
            self.trainingFolder = folder
            self.statusLabel.setText(f'Training Folder Selected: {folder}')
        else:
            self.statusLabel.setText('No folder selected')

    def runTraining(self):
        model_size = self.modelSizeComboBox.currentText()
        epochs = self.epochsLineEdit.text()
        self.model_name = self.modelNameLineEdit.text()  # Get the model name from the QLineEdit

        if not hasattr(self, 'trainingFolder') or not self.trainingFolder:
            QMessageBox.warning(self, 'Missing Information', 'Please select a training folder.')
            return
        if not epochs.isdigit():
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid number of epochs.')
            return
        if not self.model_name:
            QMessageBox.warning(self, 'Missing Information', 'Please enter a model name.')
            return

        parent_folder = os.path.dirname(self.trainingFolder)
        model_filename = f'yolov8{model_size}-cls.pt'
        command = f'yolo task=classify mode=train model={model_filename} data={self.trainingFolder} epochs={epochs} imgsz=640 name={self.model_name}'

        threading.Thread(target=lambda: subprocess.run(f'cd {parent_folder} && {command}', shell=True), daemon=True).start()
        self.statusLabel.setText('Training started...')


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = TrainingApp()
    win.show()
    sys.exit(app.exec_())
