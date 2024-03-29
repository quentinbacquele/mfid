import os
import threading
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox, QLineEdit
from PyQt5.QtCore import Qt
import glob  
import subprocess
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit
from mfid.identity.training_identity import TrainingApp


class IdentityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Monkey Identity Detection')
        self.resize(700, 100)
        
        layout = QVBoxLayout()

        self.trainingButton = QPushButton('Training', self)
        self.trainingButton.clicked.connect(self.openTrainingWindow)  
        self.loadFileButton = DarkButton('Load Videos/Images', self.openFileNameDialog)
        self.runButton = DarkButton('Run Detection', self.runDetection)
        self.statusLabel = QLabel('Status: Waiting for input', self)
        
        layout.addWidget(self.trainingButton)
        layout.addWidget(self.loadFileButton)
        layout.addWidget(self.runButton)
        layout.addWidget(self.statusLabel)
        
        self.setLayout(layout)


    def openTrainingWindow(self):
        self.trainingWindow = TrainingApp()
        self.trainingWindow.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos/Images", "", "Video Files (*.mp4 *.avi *.mov);;Image Files (*.jpg *.jpeg *.png)", options=options)
        self.filePaths = files

        if self.filePaths:
            fileNames = ', '.join([os.path.basename(f) for f in self.filePaths])
            self.statusLabel.setText(f'Files Selected: {fileNames}')
        else:
            self.statusLabel.setText('No file selected')

    def runDetection(self):
        if not hasattr(self, 'filePaths') or not self.filePaths:
            QMessageBox.warning(self, 'Missing Information', 'Please select a video or an image.')
            return
        threading.Thread(target=self.processFiles, args=(self.filePaths,), daemon=True).start()

    def processFiles(self, file_paths):
        for file_path in file_paths:
            cumulative_scores = {}
            self.processDetection(file_path)
            self.saveCumulativeResults(cumulative_scores, file_path)

    def processDetection(self, file_path, cumulative_scores):
        self.statusLabel.setText('Running detection...')
        QApplication.processEvents()  # Ensure the GUI updates the label text

        # Determine if file is an image or a video
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
        video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.MP4', '.AVI']
        file_extension = os.path.splitext(file_path)[1].lower()

        is_video = file_extension in video_extensions
        is_image = file_extension in image_extensions

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'models'))

        path_to_face_model = os.path.join(models_dir, 'best_l_face.pt')
        path_to_identity_model = os.path.join(models_dir, 'best_m_id.pt')

        face_model = YOLO(path_to_face_model)
        identity_model = YOLO(path_to_identity_model)

        save_folder = os.path.join(os.path.dirname(file_path), 'detection')
        os.makedirs(save_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        txt_file_path = os.path.join(save_folder, f"{base_name}_identity.txt")
        txt_file = open(txt_file_path, "w")
        txt_file.write("Confs\tNames\n")  # Write the header of the columns

        if is_video:
            output_file_name = f"{base_name}_identity.mp4"  # Force .avi extension for videos
            output_file_path = os.path.join(save_folder, output_file_name)

            cap = cv2.VideoCapture(file_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(3*0.5))
            frame_height = int(cap.get(4*0.5))

            fourcc = cv2.VideoWriter_fourcc(*'X264')
            out = cv2.VideoWriter(output_file_path, fourcc, frame_rate, (frame_width, frame_height))

        face_results = face_model(file_path, stream=is_video)  # Use stream mode for videos
        face_counter = 0
        for face_result in face_results:
            if is_video:
                frame = face_result.orig_img
            else:
                frame = cv2.imread(file_path)  # Read the image file directly if it's an image

            for box in face_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_img = frame[y1:y2, x1:x2]
                cropped_img_path = os.path.join(save_folder, f'face_{face_counter}.jpg')
                cv2.imwrite(cropped_img_path, cropped_img)

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

                    # Draw rectangle and text on the original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f'{conf_name}: {conf1:.2f}'
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    

                face_counter += 1

            if is_video:
                out.write(frame)  # Write out frame to video
            elif is_image:  # Save annotated image if input is an image
                annotated_image_path = os.path.join(save_folder, f"{base_name}_annotated.jpg")
                cv2.imwrite(annotated_image_path, frame)
                break  # Exit loop after processing single image

        if is_video:
            # Release everything when job is finished for videos
            out.release()
            cap.release()
        
        txt_file.close()

        # Remove cropped images and annotated frames
        files_to_remove = glob.glob(os.path.join(save_folder, 'face_*.jpg')) 
        for file in files_to_remove:
            os.remove(file)

        self.statusLabel.setText('Detection completed.')

    def saveCumulativeResults(self, cumulative_scores, file_path):
        # Calculate the mean accuracy for each label
        mean_scores = {label: sum(scores) / len(scores) for label, scores in cumulative_scores.items()}
        # Sort the results by highest mean score
        sorted_mean_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        save_folder = os.path.join(os.path.dirname(file_path), 'detection')
        os.makedirs(save_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        summary_file_path = os.path.join(save_folder, f"{base_name}_summary.txt")

        with open(summary_file_path, "w") as f:
            f.write("Label\tMean Accuracy\n")
            for label, mean_score in sorted_mean_scores:
                f.write(f"{label}\t{mean_score:.4f}\n")


  

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = IdentityApp()
    win.show()
    sys.exit(app.exec_())
