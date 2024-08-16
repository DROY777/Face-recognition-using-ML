import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from simple_facerec import SimpleFacerec
import shutil
import os
import sys  # Import the sys module for system operations

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        # Initialize OpenCV camera
        self.cap = cv2.VideoCapture(0)

        # Set camera resolution to 1920x1080 if supported (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Initialize SimpleFacerec for face recognition
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("E:/IITM/Project/Face Recognition.py/image")

        # Create GUI elements
        self.create_widgets()

        # Start video stream processing
        self.process_video_stream()

    def create_widgets(self):
        # Create a frame for displaying video feed
        self.video_frame = tk.LabelFrame(self.root, text="Video Feed")
        self.video_frame.pack(padx=10, pady=10)

        # Create a label for displaying the video feed
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Create a button for uploading an image
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side="left", padx=20)

        # Create a button for capturing an image
        self.capture_button = tk.Button(self.root, text="Capture", command=self.capture_image)
        self.capture_button.pack(side="left", padx=10)

        # Create a button for rebooting the application
        self.reboot_button = tk.Button(self.root, text="Reboot", command=self.reboot_application)
        self.reboot_button.pack(side="right",padx=30,pady=10)

    def process_video_stream(self):
        ret, frame = self.cap.read()

        if ret:
            # Resize frame to 1920x1080 for display
            frame = cv2.resize(frame, (1000,550))

            # Detect Faces
            face_locations, face_names = self.sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # Convert frame to RGB format for tkinter display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update video label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Repeat every 10 milliseconds
        self.video_label.after(10, self.process_video_stream)

    def upload_image(self):
        # Open a file dialog for image selection
        file_path = filedialog.askopenfilename()

        if file_path:
            # Ask for name to save the image with
            name = simpledialog.askstring("Input", "Enter name for the image:")

            if name:
                # Ensure the name is valid (no special characters or spaces)
                name = name.replace(" ", "_")  # Replace spaces with underscores
                name = ''.join(filter(str.isalnum, name))  # Remove special characters

                # Copy the selected image to the target directory with the entered name
                image_name = name + os.path.splitext(file_path)[1]  # Keep the original file extension
                target_path = os.path.join("E:/IITM/Project/Face Recognition.py/image", image_name)
                shutil.copyfile(file_path, target_path)

                # Load the selected image
                image = cv2.imread(target_path)

                # Detect Faces
                face_locations, face_names = self.sfr.detect_known_faces(image)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)

                # Display the processed image
                cv2.imshow("Uploaded Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def capture_image(self):
        # Capture a frame from the video feed
        ret, frame = self.cap.read()

        if ret:
            # Ask for name to save the captured image with
            name = simpledialog.askstring("Input", "Enter name for the captured image:")

            if name:
                # Ensure the name is valid (no special characters or spaces)
                name = name.replace(" ", "_")  # Replace spaces with underscores
                name = ''.join(filter(str.isalnum, name))  # Remove special characters

                # Save the captured image to the target directory with the entered name
                image_name = name + ".jpg"  # Save as JPEG format
                target_path = os.path.join("E:/IITM/Project/face/image", image_name)
                cv2.imwrite(target_path, frame)

                # Display the captured image
                cv2.imshow("Captured Image", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def reboot_application(self):
        # Restart the Python interpreter
        python = sys.executable
        os.execl(python, python, *sys.argv)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

    # Release the camera and close all OpenCV windows
    app.cap.release()
    cv2.destroyAllWindows()
