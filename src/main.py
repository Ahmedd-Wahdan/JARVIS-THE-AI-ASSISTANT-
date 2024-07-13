import pyttsx3
import requests
import speech_recognition as sr
import webbrowser
import tkinter as tk
import cv2
import threading
import face_recognition
import pywhatkit
import ultralytics
from ultralytics import YOLO
import torch 
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from supervision.draw.color import ColorPalette
from PIL import Image, ImageTk
import dlib
import time
import numpy as np
import os
from pydub import AudioSegment
from pydub.playback import play
import imageio
from tkinter import PhotoImage


class FaceRecognitionApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Assistant with Face Recognition")
        self.root.geometry("1280x720")  
        self.robot_label = tk.Label(root, text="🤖", font=("Arial", 100),fg="black")
        self.robot_label.pack(pady=20)
        self.capture = None
        self.r = sr.Recognizer()
        self.recognized_gender="female"
        self.output_text = tk.Text(self.root, height=6, width=40)
        self.output_text.pack(pady=10)
        #pil_image = Image.open(bg_image_path)
        #tk_image = ImageTk.PhotoImage(pil_image)

        # Store tk_image as an attribute

        # Create a canvas with the background image
        self.verification_textbox = tk.Text(self.root, height=2, width=40)
        self.verification_textbox.pack(pady=10)
    
        # Create a Label for displaying the GIF
        self.verification_label = tk.Label(self.root)
        self.verification_label.pack(pady=10)
        
        # Load the animated GIF frames using Pillow
        self.verification_gif_path = r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\loading-icon-animated-gif-19-1.gif"

        self.assistant_status_textbox = tk.Text(self.root, height=2, width=40)
        self.assistant_status_textbox.pack(pady=10)

        self.root.bind("<<UpdateUI2>>", lambda event=None: self.update_ui_2())
        
        # a label for the back camera
        self.camera_label = tk.Label(root)
        self.camera_label.pack()
        self.stop_back_camera = threading.Event()
        self.back_camera_thread = None 
        self.recognized=False
        self.dms_stop_event = threading.Event()
        self.dms_thread = threading.Thread(target=self.start_dms)

        self.speaking_flag = False

        self.alert_sound = AudioSegment.from_mp3(r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\alert_sound.wav")
        self.playback = None

        # binding the window closing to a function that stops the dms camera
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start the face recognition and voice assistant
        threading.Thread(target=self.start_face_recognition).start()



    def show_camera_feed(self):
        while True:
            if self.capture is not None:
                ret, frame = self.capture.read()
                
                if not ret:
                    break
                
                # Resize the frame to fit the Tkinter window
                frame = cv2.resize(frame, (1000, 800))
        
                # Convert the OpenCV frame to Tkinter PhotoImage
                tk_image = self.convert_opencv_to_tk(frame)
        
                # Update the label with the new image
                self.camera_label.config(image=tk_image)
                self.camera_label.image = tk_image
        
                if self.recognized==True and self.capture is not None:
                    self.camera_label.image = None
                    break


    def open_camera_and_predict_face(self):
        self.verification_textbox.delete(1.0, tk.END)
        self.verification_textbox.insert(tk.END, "Verifying...")
        gif_viewer = AnimatedGIFViewer(self.root, self.verification_gif_path)
        
        # Load the image of the person to be recognized
        known_image_path = r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\my_image_cropped.jpg"
        known_image = face_recognition.load_image_file(known_image_path)
        known_encoding = face_recognition.face_encodings(known_image)[0]
        known_gender = 'male'
        known_person = {'encoding': known_encoding, 'gender': known_gender}

        self.capture = cv2.VideoCapture(0)
        gif_viewer.stop()
        threading.Thread(target=self.show_camera_feed).start()

        while True:
            ret, frame = self.capture.read()
            # Find all face locations in the current frame
            face_locations = face_recognition.face_locations(frame)
            
            if face_locations:
                # Encode the first face found in the frame
                current_encoding = face_recognition.face_encodings(frame, face_locations)[0]

                # Compare the current face encoding with the known encoding
                results = face_recognition.compare_faces([known_person['encoding']], current_encoding)
        
                if results[0]:
                    self.recognized=True
                    recognized_gender = known_person['gender']
                    self.recognized_gender=recognized_gender
                    self.verification_textbox.delete(1.0, tk.END)
                    self.verification_textbox.insert(tk.END, f"Verified - Gender: {recognized_gender}")
                    self.assistant_status_textbox.insert(tk.END, "Assisting...")
                    break

        threading.Thread(target=self.start_voice_assistant).start()
        self.dms_thread.start()

    def start_voice_assistant(self):
        gif_viewer = None
        
        with sr.Microphone() as mic:
            while True:
                if gif_viewer:
                    gif_viewer.stop()
                self.camera_label.config(image=None)
                gif_viewer = AnimatedGIFViewer(self.root, r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\bddf8a11582713.560fa0db0dee5.gif")
                try:
                    self.root.event_generate("<<UpdateUI2>>")
                    self.speak("jarvis is listening for the wake word",self.recognized_gender)
                    self.r.adjust_for_ambient_noise(mic, duration=0.2)
                    audio = self.r.listen(mic,timeout=4,phrase_time_limit=4)
                    
                    text = self.r.recognize_google(audio)
                    text = text.lower()
                    if "hey jarvis" in text:
                        gif_viewer.stop()
                        self.root.after(100, self.update_ui_color)
                        self.speak("what can i do for you today sir",self.recognized_gender)
                        audio = self.r.listen(mic,timeout=4,phrase_time_limit=4)
                        text = self.r.recognize_google(audio)
                        text = text.lower()
                        
                        if text is not None:
                            self.root.after(100, self.update_ui_1,text)
                            if "how are you" in text:
                                self.speak("Never been better, master wahhhdan",self.recognized_gender)
                            elif "hello" in text:
                                self.speak("Hi there.",self.recognized_gender)
                            elif "open google" in text:
                                webbrowser.open_new('http://google.com')
                            elif "go to sleep" in text:
                                self.speak("Okay, goodnight sir.",self.recognized_gender)
                                self.on_close()
                                exit()
                            
                            elif "google" in text:
                                google_index = text.index("google")
                                search_query = text[google_index + len("google"):].strip()
                                pywhatkit.search(search_query)
                                
                            elif "youtube" in text:
                                google_index = text.index("youtube")
                                search_query = text[google_index + len("youtube"):].strip()
                                pywhatkit.playonyt(search_query)

                            elif "spotify" in text:
                                spotify_index = text.index("spotify")
                                search_query = text[spotify_index + len("spotify"):].strip()
                                self.play_song_on_spotify_app(search_query)
                            
                            elif "camera" in text and "stop" in text:
                                self.close_back_camera()
                                self.speak("Stopping back camera",self.recognized_gender)
                            
                            elif "camera" in text and "open" in text:
                                if not self.back_camera_thread or not self.back_camera_thread.is_alive():
                                    self.back_camera_thread = threading.Thread(target=self.open_back_camera)
                                    self.back_camera_thread.start()
                                else:
                                    self.speak("The back camera is already active.",self.recognized_gender)
                                
                            elif "weather" in text:
                                self.get_and_speak_weather()
                            
                            else:
                                self.speak("I'm sorry, sir. I did not understand your request.",self.recognized_gender)

                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    continue
                except sr.exceptions.WaitTimeoutError:
                    continue

    def start_face_recognition(self):
        # Start face recognition before voice  assistant
        self.open_camera_and_predict_face()

        
    def update_ui_1(self,text):
        # This function updates the UI from the main thread
        self.output_text.insert(tk.END, text+"\n")
        
    def update_ui_color(self):
        # This function updates the UI from the main thread
        self.robot_label.config(fg="red")
    
    def update_ui_2(self):
        # This function updates the UI from the main thread
        self.robot_label.config(fg="black")
        
    
    def speak(self, text, gender):
        if self.speaking_flag:
            return       

        self.speaking_flag = True

        rate = 100
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Set voice based on gender
        if gender.lower() == 'male':
            engine.setProperty('voice', voices[0].id)  # Assuming the first voice is male
        elif gender.lower() == 'female':
            engine.setProperty('voice', voices[1].id)  # Assuming the second voice is female

        engine.setProperty('rate', rate + 75)
        engine.say(text)
        engine.runAndWait()

        self.speaking_flag = False

    def play_song_on_spotify_app(self, song_name):
            search_url = f"spotify:search:{song_name}"
            os.system(f"start {search_url}")
    
    def load_yolo(self, model_path): 
        model = YOLO(model_path)
        model.fuse()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model

    def open_back_camera(self):
        model = self.load_yolo(r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\CX_Deployment\yolov8s.pt")

        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
        
        # dict maping class_id to class_name
        CLASS_NAMES_DICT = model.model.names
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

        # Use if the voice command didn't work
        self.root.bind_all("<KeyPress-q>", self.close_back_camera)
        
        while not self.stop_back_camera.is_set() and self.back_camera_thread.is_alive():
            # Read a frame from the camera
            ret, frame = self.capture.read()
        
            if not ret or self.stop_back_camera.wait(timeout=0.07):
                break
            
            # Perform inference on the GPU
            results = model(frame)
            detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            
            # format custom labels
            labels = [
                f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
        
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        
            # Resize the frame to fit the Tkinter window
            frame = cv2.resize(frame, (800, 600))

            # Convert the OpenCV frame to Tkinter PhotoImage
            tk_image = self.convert_opencv_to_tk(frame)

            # Update the label with the new image
            self.camera_label.config(image=tk_image)
            self.camera_label.image = tk_image
        
            self.root.update_idletasks()

        
    def convert_opencv_to_tk(self, frame):
            """Convert OpenCV image to Tkinter PhotoImage."""
            b, g, r = cv2.split(frame)
            img = cv2.merge((r, g, b))
            img = Image.fromarray(img)
            tk_image = ImageTk.PhotoImage(image=img)
            return tk_image

    def close_back_camera(self, event=None):
        if self.back_camera_thread and self.back_camera_thread.is_alive():
            # Set the stop_back_camera flag to signal thread termination
            self.stop_back_camera.set()
            # Wait for the thread to finish before continuing
            self.back_camera_thread.join()
            # Reset the flag for potential future use
            self.stop_back_camera.clear()
            # Schedule the cleanup actions in the main thread using after
            self.root.after(10, self.cleanup_back_camera)

    def cleanup_back_camera(self):
        self.camera_label.config(image=None)
        self.camera_label.image = None
        

    def start_dms(self):
        landmarks_window, landmarks_label = self.create_landmarks_window()
        
        # Load the facial landmark predictor
        predictor = dlib.shape_predictor(r"C:\Users\ahmed\Desktop\my projects\JARVIS_GITHUB\JARVIS_V0.1\CX_Deployment\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
        face_detector = dlib.get_frontal_face_detector()
        
        # Threshold for blink detection
        EAR_THRESHOLD = 0.25

        self.capture = cv2.VideoCapture(0)
        
        # Blink timer variables
        blink_start_time = 0
        blink_duration = 0
        blink_duration_first_threshold = 3
        blink_duration_second_threshold = 5
        first_alert_given = False
        
        while not self.dms_stop_event.is_set() and self.dms_thread.is_alive():
            ret, frame = self.capture.read()

            if not ret or self.dms_stop_event.is_set():
                break
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Use dlib to detect faces
            faces = face_detector(gray)
            
            for face in faces:
                # Get facial landmarks
                landmarks = predictor(gray, face)
                landmarks = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])

                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
                # Update the landmarks window
                landmarks_window.update_idletasks()
                
                # Extract left and right eye coordinates
                left_eye = landmarks[42:48]
                right_eye = landmarks[36:42]
        
                # Compute eye aspect ratios
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                    
                # Check for blinks
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    if blink_start_time == 0:
                        blink_start_time = time.time()
                    blink_duration = time.time() - blink_start_time
                else:
                    blink_start_time = 0
                    blink_duration = 0
                    first_alert_given = False
        
                if blink_duration >= blink_duration_first_threshold and not first_alert_given:
                    self.speak("You may be tired, Consider taking a break", self.recognized_gender)
                    first_alert_given = True
        
                if blink_duration >= blink_duration_second_threshold:
                    play(self.alert_sound)

            resized_frame = cv2.resize(frame, (600, 600))
            
            # Convert the frame to Tkinter PhotoImage
            tk_image = self.convert_opencv_to_tk(resized_frame)

            # Update the label with the new image
            landmarks_label.config(image=tk_image)
            landmarks_label.image = tk_image
        
        if not self.dms_thread.is_alive():
            # Release the webcam and close the window
            landmarks_window.destroy()
            self.capture.release()
            cv2.destroyAllWindows()
        
    def get_and_speak_weather(self):
        # Function to get weather information and speak it to the user
        api_key = '6ba0421f5c194ad2a15183722241003'  
        city_name = 'Cairo' 

        base_url = "https://api.weatherapi.com/v1/current.json"
        params = {
            'key': api_key,
            'q': city_name,
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            if 'error' in data:
                print(f"Error: {data['error']['message']}")
                return

            temperature = data['current']['temp_c']
            condition = data['current']['condition']['text']

            # Speak the weather details
            weather_message = f"The current temperature in {city_name} is {temperature}°C, and the weather condition is {condition}."
            self.speak(weather_message, self.recognized_gender)

        except requests.RequestException as e:
            print(f"Request Error: {e}")




    def eye_aspect_ratio(self, eye):
        # Calculate Euclidean distances between the two sets of vertical landmarks (y-coordinates)
        a = np.linalg.norm(eye[1] - eye[5])
        b = np.linalg.norm(eye[2] - eye[4])
    
        # Calculate Euclidean distance between the horizontal landmarks (x-coordinates)
        c = np.linalg.norm(eye[0] - eye[3])
    
        # Compute the eye aspect ratio
        ear = (a + b) / (2.0 * c)
        return ear

    def on_close(self):
        if self.dms_thread.is_alive():
            # Set the flag to signal the DMS thread to exit
            self.dms_stop_event.set()
            
            # Stop the DMS thread if it's running
            self.dms_thread.join()

        # Release the camera resources
        if self.capture:
            self.capture.release()

        # Close the main window
        self.root.destroy()

    def create_landmarks_window(self):
        landmarks_window = tk.Toplevel(self.root)
        landmarks_window.title("Facial Landmarks")
        landmarks_window.geometry("600x600")
        landmarks_label = tk.Label(landmarks_window)
        landmarks_label.pack()
        return landmarks_window, landmarks_label
        
class AnimatedGIFViewer:
    def __init__(self, master, gif_path):
        self.master = master
        self.frames = self.load_frames(gif_path)
        self.current_frame = 0
        self.image_label = tk.Label(self.master)
        self.image_label.pack()
        self.running = True
        self.display_frame()

    def load_frames(self, gif_path):
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = ImageTk.PhotoImage(gif.copy())
                frames.append(frame)
                gif.seek(len(frames))  # Move to the next frame
        except EOFError:
            return frames  # End of frames
    
    def display_frame(self):
        if not self.running:
            return
            
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.image_label.config(image=self.frames[self.current_frame])
        self.image_label.image = self.frames[self.current_frame]
        self.master.after(50, self.display_frame)  # Adjust the interval for frame update

    def stop(self):
        self.running = False
        self.image_label.destroy()  # Remove the label from the UI

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()  