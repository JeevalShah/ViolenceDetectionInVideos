import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import imutils
from collections import deque
import time
from IPython.display import HTML
from base64 import b64encode

# Constants
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64     
CLASSES_LIST = ["NonViolence", "Violence"]
MoBiLSTM_model = load_model('ViolenceDetectionModel.h5')

def detect_people_video(video_file_path, output_file_path, SEQUENCE_LENGTH):

    # Initialize the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # VideoWriter to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                    int(video_reader.get(cv2.CAP_PROP_FPS)), (original_video_width, original_video_height))
 
    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        ok, frame = video_reader.read() 
        if not ok:
            break

        high_confidence, moderate_confidence, low_confidence = 0, 0, 0

        # Convert the frame to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people in the resized grayscale frame
        rects, weights = hog.detectMultiScale(img_gray, padding=(4, 4), scale=1.02)

        # Draw rectangles based on confidence levels
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.13:
                continue
            elif weights[i] < 0.3 and weights[i] > 0.13:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                low_confidence += 1
            if weights[i] < 0.7 and weights[i] > 0.3:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
                moderate_confidence += 1
            if weights[i] > 0.7:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                high_confidence += 1
            
            cv2.putText(frame, 'High Confidence: ' + str(high_confidence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, 'Moderate Confidence: ' + str(moderate_confidence), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 122, 255), 2)
            cv2.putText(frame, 'Low Confidence: ' + str(low_confidence), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Write The frame into the disk using the VideoWriter
        video_writer.write(frame)                       

    video_reader.release()
    video_writer.release()

def predict_video(input_file_path):
    video_reader = cv2.VideoCapture(input_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Store the predicted class in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        success, frame = video_reader.read() 
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
 
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_list, axis = 0))[0]
 
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
 
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted class along with the prediction confidence.
    if predicted_class_name == "Violence":
        st.error(f'Predicted: {predicted_class_name}')
        st.error(f'Confidence: {predicted_labels_probabilities[predicted_label]}')
    else:
        st.success(f'Predicted: {predicted_class_name}')
        st.success(f'Confidence: {predicted_labels_probabilities[predicted_label]}')
    video_reader.release()


#Frame By Frame Prediction
def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):
    
    # Read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # VideoWriter to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
 
    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
 
    # Store the predicted class in the video.
    predicted_class_name = ''
 
    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        ok, frame = video_reader.read() 
        
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame 
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:                        

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
         
        # Write The frame into the disk using the VideoWriter
        video_writer.write(frame)                       

    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

def save(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Display a success message
    st.success(f"Video saved locally")
    
st.title("Violence Detection Website")

# File Upload
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    save(uploaded_file)

    if st.button("Detect People"):
        detect_video_file_path = "Detect_" + uploaded_file.name
        detect_people_video(uploaded_file.name, detect_video_file_path, SEQUENCE_LENGTH)

        predict_area = st.empty()
        cap = cv2.VideoCapture(detect_video_file_path)

        while cap.isOpened():
            ret, frame_detect = cap.read()

            if not ret:
            # If the video has reached the end, reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb_detect = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2RGB)
            predict_area.image(frame_rgb_detect, channels='RGB')

    # Display a header for predictions
    st.header("Predictions")
    predict_video(uploaded_file.name)

    if st.button("Show Frame By Frame Prediction"):
        predict_video_file_path = "Predict_" + uploaded_file.name
        predict_frames(uploaded_file.name, predict_video_file_path, SEQUENCE_LENGTH)

        predict_area = st.empty()
        cap = cv2.VideoCapture(predict_video_file_path)

        while cap.isOpened():
            ret, frame_predict = cap.read()

            if not ret:
            # If the video has reached the end, reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb_predict = cv2.cvtColor(frame_predict, cv2.COLOR_BGR2RGB)
            predict_area.image(frame_rgb_predict, channels='RGB')
