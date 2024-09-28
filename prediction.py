
import cv2
import time
import datetime
import numpy as np
import threading
import queue
# from send_alert import send_email
from tasks import *
###################################################  PREDICTION WORKER  ########################################################################################3
# Prediction worker thread function with warm-up phase
def predict_worker(model, frame_queue, stop_event, present_recording_prediction_list, mail_sent, filename):

    voilence_started =  False
    voilence_started_at = None

    while not stop_event.is_set():    # while stop event is not set 

        if not frame_queue.empty():   # if the queue is not empty

            frames = frame_queue.get()
            # print(f"its frame -> {len(frames)}")

            if frames is None:  # Stop signal
                break
    
            video_frames = np.expand_dims(np.array(frames), axis=0)

            # predict 
            pred_fight = model.predict(video_frames)
            if pred_fight[0][0] >= 0.6:   # threshold
                print(f"Violence detected with confidence: {pred_fight[0][0]}")
                present_recording_prediction_list.append(1)
                if not voilence_started:
                    voilence_started = True
                    voilence_started_at = str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            else:
                print(f"No violence detected. Confidence: {pred_fight[0][0]}")
                present_recording_prediction_list.append(0)

             # sending alert mail for each recording at real time, asynchrously using celery tasks
            if len(present_recording_prediction_list) == 2 and not mail_sent:  ########## change to 12
                results = present_recording_prediction_list.copy()
                # print(results)
                results = np.array(results)
                print(results) ##########

                # if 6 out of past 10 predictions have violence send email asynchrously
                if results.sum() >= 1: ################## change to 8
                    print("ok results")
                    mail_sent =  True

                    send_email.delay("violencedetectiona@gmail.com", "VIOLENCE ALERT", f"There is voilence in {filename}.mp4 at {voilence_started_at}!!!!")

                    print(f"MAIL there is voilence in {filename}.mp4 at {voilence_started_at}!!!!!!!!!!")
                # removing oldest prediction
                present_recording_prediction_list = present_recording_prediction_list[1:]
                
        else:
            time.sleep(0.1)  # wait for frames

def start_prediction_thread(model, frame_queue, stop_event, filename):
    present_recording_prediction_list = []
    mail_sent = False
    prediction_thread = threading.Thread(target=predict_worker, args=(model, frame_queue, stop_event, present_recording_prediction_list, mail_sent, filename))
    prediction_thread.start()
    return prediction_thread
