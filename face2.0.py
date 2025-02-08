from facenet_pytorch import MTCNN
import torch
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import mtcnn

# Define the device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define our MTCNN extractor with GPU support
fast_mtcnn = MTCNN(
    image_size=160,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

# Function to run face detection and record performance metrics
def run_detection(fast_mtcnn, filenames, batch_size=60, skip_frames=5):
    frames_processed = 0
    faces_detected = 0
    start = time.time()
    
    faces_count = []
    frames_count = []
    times = []

    for filename in tqdm(filenames):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for j in range(v_len):
            ret, frame = v_cap.read()
            if not ret:
                break

            # Process every 'skip_frames' frame to speed up detection
            if j % skip_frames == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize frame to reduce computation
                frames.append(frame)

            # Run face detection in batches
            if len(frames) >= batch_size or j == v_len - 1:
                faces = fast_mtcnn(frames)
                frames_processed += len(frames)
                faces_detected += len(faces) if faces else 0
                frames = []

                # Record performance metrics
                current_time = time.time() - start
                faces_count.append(faces_detected)
                frames_count.append(frames_processed)
                times.append(current_time)

                # Print progress and performance metrics
                print(
                    f'Frames per second: {frames_processed / current_time:.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )
        v_cap.release()
    
    # Plot the performance metrics using Matplotlib
    plot_metrics(times, frames_count, faces_count)

# Function to plot performance metrics using Matplotlib
def plot_metrics(times, frames_count, faces_count):
    plt.figure(figsize=(10, 5))

    # Plot frames processed over time
    plt.subplot(1, 2, 1)
    plt.plot(times, frames_count, label='Frames Processed', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frames Processed')
    plt.title('Frames Processed Over Time')
    plt.grid(True)
    plt.legend()

    # Plot faces detected over time
    plt.subplot(1, 2, 2)
    plt.plot(times, faces_count, label='Faces Detected', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Faces Detected')
    plt.title('Faces Detected Over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to display image and detect faces
def display_image_with_faces(filename, detector):
    # Load the image using OpenCV
    image = cv2.imread(filename)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display with Matplotlib

    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_image)

    # Plot the original image with Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title("Original Image with Detected Faces")
    plt.axis('off')

    # Draw bounding boxes and keypoints on the image
    for face in faces:
        x, y, width, height = face['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green', linewidth=2)
        plt.gca().add_patch(rect)

        for key, value in face['keypoints'].items():
            plt.plot(value[0], value[1], marker='o', markersize=5, color='orange')

    # Show the modified image
    plt.show()

# Example filenames for detection
filenames = ['vid1.mp4', 'vid2.mp4']  # Replace with actual filenames
run_detection(fast_mtcnn, filenames)

# Example of displaying an image and detecting faces
filename = 'image.jpg'  # Replace with actual filename
detector = mtcnn.MTCNN()  # Initialize MTCNN detector
display_image_with_faces(filename, detector)
