import cv2
import time
import datetime
import tensorflow as tf
import threading
import queue
from flask import Flask
# import smtplib
# from email.mime.text import MIMEText
import numpy as np
# from flask_mail import Mail, Message
from prediction import *
from model import VD_model


def create_app():
    app = Flask(__name__)
    # app.config.update(CELERY_BROKER_URL='redis://localhost:6379',CELERY_RESULT_BACKEND='redis://localhost:6379')
    return app

app = create_app()

###################################################  MAIN  ##################################################################################
# Main video capture and processing
def main():
    # Initialize the model
    model = VD_model(tf, "final_model12.weights.h5")

    # creating HOG descriptor to detect humans
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # initializing variables for multiplethreading to concurrently capture images and predict Voilence
    prediction_thread = None
    stop_event = threading.Event()  # Event to signal thread to stop

     # queue to store frames for prediction
    frame_queue = queue.Queue(maxsize=16)

    # initializing variables
    recording = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 10

    # input size of image give to model
    target_height = 160
    target_width = 160

    # creating img capturing obj
    cap = cv2.VideoCapture(0)
    
    # Initialize face & body detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Frame size and codec
    frame_size = (int(cap.get(3)), int(cap.get(4))) # fteching height and width of the captured images from the webcam
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frames = [] # list of frames provided to the model

    while True: # webcam will keep collecting images in an infinite loop until "q" is pressed

        ret, frame = cap.read()

        if not ret:
            break # if no image is captured break the while loop

        # Convert to grayscale for face/body detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        # bodies = body_cascade.detectMultiScale(gray, 1.5, 5)
        
        # detecting humans with HOG
        (rects, _ ) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Prepare the frame for prediction
        frame_p = cv2.resize(frame, (target_width, target_height)) # resize
        frame_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2RGB)         # convert to RGB for VGG19
        # scaling is not required as vgg19 requires input pixel values between 0-255
      

        # putting time in frames to be recorded
        dt = str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, dt, (frame_size[0]-270, frame_size[1]-25), font, 0.5, (255, 255, 255), 2, cv2.LINE_8) 
                                        # position                                font scale, color,    thickness

        # Detection Logic
        if len(rects) + len(faces) > 0:  # if humans are detected by the HOG

            # appending each frame to the list frames to be sent for prediction
            frames.append(frame_p)

            if len(frames) == 16:  # Once we have 16 frames send them for prediction
                if not frame_queue.full():
                #     print("Dropping frames due to processing delay!")
                # else:
                    frame_queue.put(frames.copy())  # if count of frames == 16, put frames in queue for prediction
                    frames.clear() # empty the frames list

            if recording:  # when its still recording
                timer_started = False  # disable the timer
            else:  # humans are dtected but recording has not started
                recording = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)  
                print("Started Recording!")

                # start prediction thread when recording starts
                if prediction_thread is None or not prediction_thread.is_alive():
                    stop_event.clear()  # clear stop signal
                    prediction_thread = start_prediction_thread(model, frame_queue, stop_event, current_time)

        elif recording:  # humans are not detected while its recording
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:  # if humans are not recorded for a while stop recording
                    recording = False
                    timer_started = False
                    out.release()
                    print("Stopped Recording!")
                    frames.clear()

                    # Stop the prediction thread when recording stops
                    stop_event.set()  # Signal the worker to stop
                    if prediction_thread is not None:
                        prediction_thread.join()
                        prediction_thread = None  # Reset thread reference
            else:    # start the timer
                timer_started = True
                detection_stopped_time = time.time()
                frames.append(frame_p)

                if len(frames) == 16:  # Once we have 16 frames, send them for prediction
                    if not frame_queue.full():
                    #     print("Dropping frames due to processing delay!")
                    # else:
                        frame_queue.put(frames.copy())  # Put frames in queue for prediction
                        frames.clear()

        # if recording save frames to video file
        if recording:
            out.write(frame)

        cv2.imshow("Camera", frame) ###############

        if cv2.waitKey(1) == ord('q'):  # press "q" to stop webcam and exit while loop
            break

    # if webcam is not collecting frames stop the prediction thread
    if prediction_thread is not None:
        stop_event.set()  # signal the worker to exit
        prediction_thread.join()

    cap.release()
    cv2.destroyAllWindows()

main()
###########################################################################################################################################

if __name__=="__main__":
    app.run()

