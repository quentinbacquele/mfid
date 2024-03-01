import os
import threading
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLabel
from PyQt5.QtCore import Qt
import glob  
from mfid.utils.theme_dark import DarkTheme, DarkButton, DarkLineEdit

class IdentityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        DarkTheme(self)
        self.setWindowTitle('Monkey Identity Detection')
        self.resize(500, 200)
        
        layout = QVBoxLayout()
        
        self.loadFileButton = DarkButton('Load Video/Image', self.openFileNameDialog)
        self.runButton = DarkButton('Run Detection', self.runDetection)
        self.statusLabel = QLabel('Status: Waiting for input', self)
        
        layout.addWidget(self.loadFileButton)
        layout.addWidget(self.runButton)
        layout.addWidget(self.statusLabel)
        
        self.setLayout(layout)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Video/Image", "", "All Files (*);;Video Files (*.mp4 *.avi *.mov);;Image Files (*.jpg *.jpeg *.png)", options=options)
        if file:
            self.fileName = file
            self.statusLabel.setText('File Selected: ' + os.path.basename(file))

    def runDetection(self):
        if not hasattr(self, 'fileName'):
            QMessageBox.warning(self, 'Missing Information', 'Please select a video or an image.')
            return
        threading.Thread(target=self.processDetection, args=(self.fileName,), daemon=True).start()

    def processDetection(self, file_path):
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

        if is_video:
            output_file_name = f"{base_name}_identity.avi"  # Force .avi extension for videos
            output_file_path = os.path.join(save_folder, output_file_name)

            cap = cv2.VideoCapture(file_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

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
                    idx = identity_result.probs.top1
                    conf = identity_result.probs.top1conf.item()
                    names_dict = identity_result.names
                    conf_name = names_dict[idx]

                    # Draw rectangle and text on the original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f'{conf_name}: {conf:.2f}'
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

        # Remove cropped images and annotated frames
        files_to_remove = glob.glob(os.path.join(save_folder, 'face_*.jpg')) 
        for file in files_to_remove:
            os.remove(file)

        self.statusLabel.setText('Detection completed.')


    def closeEvent(self, event):
        QApplication.quit()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = IdentityApp()
    win.show()
    sys.exit(app.exec_())
