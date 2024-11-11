import cv2
import numpy as np
from PIL import Image
import io
from tkinter import Tk, filedialog

# Load the face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces_in_image(file_path):
    # Load the image using PIL
    pil_image = Image.open(file_path).convert("RGB")
    open_cv_image = np.array(pil_image)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces and count them
    face_count = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_count += 1

    # Display the number of faces detected
    print(f"Number of faces detected: {face_count}")

    # Display the result image with rectangles
    display_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(display_image).show()

# Set up Tkinter to get the file path from the user
root = Tk()
root.withdraw()  # Hide the root window

# Open a file dialog to select an image
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Run face detection on the selected image
if file_path:
    detect_faces_in_image(file_path)
else:
    print("No file selected.")
