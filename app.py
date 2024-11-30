# import cv2
# import numpy as np
# import pyttsx3
# import speech_recognition as sr
# from threading import Thread, Lock, Event
# from queue import PriorityQueue
# from ultralytics import YOLO
# import time

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 220)  # Increase speech rate for faster announcements

# # Lock to control access to the text-to-speech engine
# tts_lock = Lock()

# # Load the YOLOv8 Nano model
# model = YOLO('yolov8n.pt')  # Changed to YOLOv8 Nano

# # Set up the recognizer and microphone for speech recognition
# recognizer = sr.Recognizer()
# microphone = sr.Microphone()

# # Flags to control object detection and exiting
# detecting = False
# exit_event = Event()
# stop_event = Event()

# # Priority queue to handle announcement order
# announce_queue = PriorityQueue()

# # To keep track of the last announcement time for each object
# last_announcement_time = {}

# def announce(text):
#     with tts_lock:
#         engine.say(text)
#         engine.runAndWait()

# def process_announcement_queue():
#     while not exit_event.is_set():
#         if not announce_queue.empty():
#             announcements = []
#             while not announce_queue.empty():
#                 priority, announcement = announce_queue.get()
#                 announcements.append(announcement)
            
#             # Batch announcement
#             if announcements:
#                 announcement_text = "; ".join(announcements)
#                 announce(announcement_text)

# def listen_for_commands():
#     global detecting
#     while not exit_event.is_set():
#         with microphone as source:
#             recognizer.adjust_for_ambient_noise(source)
#             print("Listening for commands...")
#             try:
#                 audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
#                 command = recognizer.recognize_google(audio).lower()
#                 print(f"Command received: {command}")

#                 if "start" in command:
#                     stop_event.clear()
#                     detecting = True
#                     announce_queue.put((0, "Starting detection."))
#                 elif "stop" in command:
#                     detecting = False
#                     stop_event.set()
#                     announce_queue.put((0, "Stopping detection."))
#                     announce("Detection stopped.")
#                 elif "exit" in command:
#                     detecting = False
#                     stop_event.set()
#                     exit_event.set()
#                     announce_queue.put((0, "Exiting Dristi."))
#                     announce("Exiting Dristi.")
#                     break
#             except sr.WaitTimeoutError:
#                 print("Listening timeout; no command heard.")
#             except sr.UnknownValueError:
#                 print("Could not understand the command.")
#             except sr.RequestError as e:
#                 print(f"Could not request results; network error: {e}")
#                 announce_queue.put((0, "Network error. Please check your connection."))

# def calculate_distance(box, frame_width):
#     box_width = box[2] - box[0]
#     relative_size = box_width / frame_width
#     distance = 1 / relative_size  # Inverse relation
#     return round(distance, 2)

# def detect_objects():
#     global detecting

#     cap = cv2.VideoCapture(0)
#     frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     previous_positions = {}
#     previous_distances = {}
    
#     # Track last announced state
#     last_announced_states = {}

#     # Set thresholds
#     movement_threshold = 30  # Increased threshold for movement detection
#     confidence_threshold = 0.5  # Lowered for YOLOv8 Nano, adjust as necessary

#     while cap.isOpened() and not exit_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Display the real-time video feed
#         cv2.imshow('Real-Time Object Detection', frame)

#         if detecting:
#             results = model(frame)
#             current_positions = {}
#             detected_objects = {"closer": [], "farther": [], "unidentified": []}

#             for result in results[0].boxes:
#                 if stop_event.is_set():
#                     break
                
#                 box = result.xyxy[0].cpu().numpy().astype(int)
#                 confidence = result.conf.item()
                
#                 if confidence < confidence_threshold:
#                     class_name = "unidentified object"
#                 else:
#                     class_name = model.names[int(result.cls.item())]
                
#                 center_x = (box[0] + box[2]) // 2
#                 center_y = (box[1] + box[3]) // 2

#                 current_position = np.array([center_x, center_y])
#                 distance = calculate_distance(box, frame_width)
#                 obj_id = result.cls.item()

#                 current_time = time.time()

#                 # Debounce announcements for the same object within 15 seconds
#                 if obj_id not in last_announcement_time or (current_time - last_announcement_time[obj_id]) > 15:
#                     if obj_id not in previous_positions:
#                         previous_positions[obj_id] = current_position
#                         previous_distances[obj_id] = distance
#                         last_announced_states[obj_id] = None
#                         last_announcement_time[obj_id] = current_time
#                         if class_name == "unidentified object":
#                             detected_objects["unidentified"].append(f"{class_name} detected at a distance of {distance} meters.")
#                         else:
#                             detected_objects["closer"].append(f"{class_name} detected at a distance of {distance} meters.")
#                     else:
#                         # Calculate movement vector
#                         movement_vector = current_position - previous_positions[obj_id]
#                         movement_distance = np.linalg.norm(movement_vector)

#                         # Determine direction of movement
#                         direction = "closer" if movement_vector[1] < 0 else "farther"

#                         # Only announce if the movement direction has changed
#                         last_state = last_announced_states.get(obj_id, None)
#                         if movement_distance > movement_threshold and direction != last_state:
#                             detected_objects[direction].append(f"{class_name} is moving {direction} to you at a distance of {distance} meters.")
#                             last_announced_states[obj_id] = direction
#                             last_announcement_time[obj_id] = current_time

#                         # Update previous positions and distances
#                         previous_positions[obj_id] = current_position
#                         previous_distances[obj_id] = distance

#             # Consolidate announcements
#             for direction, objects in detected_objects.items():
#                 if objects:
#                     announcement_text = f"{', '.join(objects)}"
#                     announce_queue.put((0, announcement_text))

#         # Press 'q' to quit the real-time video feed
#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Start the announcement processing thread
#     announcement_thread = Thread(target=process_announcement_queue)
#     announcement_thread.start()

#     # Run the command listener in a separate thread
#     command_thread = Thread(target=listen_for_commands)
#     command_thread.start()

#     # Start object detection
#     detect_objects()

#     # Wait for the threads to finish
#     command_thread.join()
#     announcement_thread.join()






import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
from threading import Thread, Lock, Event
from queue import PriorityQueue
from ultralytics import YOLO
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 220)

# Lock to control access to the text-to-speech engine
tts_lock = Lock()

# Load the YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Set up the recognizer and microphone for speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Flags and queues
detecting = False
exit_event = Event()
stop_event = Event()
announce_queue = PriorityQueue()
last_announcement_time = {}
requested_objects = []  # List of requested objects to detect

def announce(text):
    """Thread-safe function to announce text using TTS."""
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

def process_announcement_queue():
    """Thread to process and announce queued messages."""
    while not exit_event.is_set():
        if not announce_queue.empty():
            _, announcement = announce_queue.get()
            announce(announcement)

def listen_for_commands():
    """Thread to listen for voice commands."""
    global detecting, requested_objects
    while not exit_event.is_set():
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for commands...")
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                if "start" in command:
                    stop_event.clear()
                    detecting = True
                    announce_queue.put((0, "Starting detection."))
                elif "stop" in command:
                    detecting = False
                    stop_event.set()
                    announce_queue.put((0, "Stopping detection."))
                    announce("Detection stopped.")
                elif "exit" in command:
                    detecting = False
                    stop_event.set()
                    exit_event.set()
                    announce_queue.put((0, "Exiting Dristi."))
                    announce("Exiting Dristi.")
                    break
                else:
                    # Add object name to the requested list
                    requested_objects.append(command)
                    announce_queue.put((0, f"Looking for {command}."))
            except sr.WaitTimeoutError:
                print("Listening timeout; no command heard.")
            except sr.UnknownValueError:
                print("Could not understand the command.")
            except sr.RequestError as e:
                print(f"Could not request results; network error: {e}")
                announce_queue.put((0, "Network error. Please check your connection."))

def calculate_distance(box, frame_width):
    """Calculate the approximate distance of an object based on its bounding box."""
    box_width = box[2] - box[0]
    relative_size = box_width / frame_width
    distance = 1 / relative_size
    return round(distance, 2)

def detect_objects():
    """Main object detection loop."""
    global detecting, requested_objects

    cap = cv2.VideoCapture(0)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if not cap.isOpened():
        announce("Camera not available.")
        return

    confidence_threshold = 0.5  # Adjust for YOLOv8 Nano
    previous_positions = {}
    previous_distances = {}
    last_announced_states = {}

    while cap.isOpened() and not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        cv2.imshow('Real-Time Object Detection', frame)

        if detecting:
            results = model(frame)
            detected_names = set()

            detected_objects = {"closer": [], "farther": [], "unidentified": []}

            for result in results[0].boxes:
                box = result.xyxy[0].cpu().numpy().astype(int)
                confidence = result.conf.item()

                if confidence >= confidence_threshold:
                    class_name = model.names[int(result.cls.item())]
                    detected_names.add(class_name)
                    distance = calculate_distance(box, frame_width)
                    
                    # Automatically announce detected objects
                    detected_objects["closer"].append(f"{class_name} detected at a distance of {distance} meters.")

            # Announce detected objects
            for direction, objects in detected_objects.items():
                if objects:
                    announcement_text = f"{', '.join(objects)}"
                    announce_queue.put((0, announcement_text))

            # Check requested objects
            for obj in requested_objects:
                if obj in detected_names:
                    announce_queue.put((0, f"{obj} is present in the frame."))
                else:
                    announce_queue.put((0, f"{obj} is not detected."))

            # Clear the requested objects list after processing
            requested_objects.clear()

        # Press 'q' to quit the video feed manually
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Start the announcement processing thread
        announcement_thread = Thread(target=process_announcement_queue, daemon=True)
        announcement_thread.start()

        # Run the command listener in a separate thread
        command_thread = Thread(target=listen_for_commands, daemon=True)
        command_thread.start()

        # Start object detection
        detect_objects()

        # Wait for threads to finish
        command_thread.join()
        announcement_thread.join()
    except KeyboardInterrupt:
        print("Program interrupted.")
        exit_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        announce(f"An error occurred: {e}")



