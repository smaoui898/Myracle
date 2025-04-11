import cv2
import time
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import playsound
import os
import threading
import queue
import tempfile
import sounddevice as sd 
import soundfile as sf 


#Voice announcement system
voice_queue = queue.Queue()
is_speaking = False
object_counter = {}  #Track detected objects


def play_audio(file_path):
    try:
        # Read audio file
        data, samplerate = sf.read(file_path)
        # Play with sounddevice for better control
        sd.play(data, samplerate)
        sd.wait()  #Wait until playback is finished
    except Exception as e:
        print(f"Audio playback error: {e}")
        try:
            #Fallback to playsound if sounddevice fails
            playsound.playsound(file_path)
        except Exception as fallback_e:
            print(f"Fallback playback failed: {fallback_e}")


def voice_worker():
    global is_speaking
    while True:
        text = voice_queue.get()
        try:
            is_speaking = True
            #Create temp file in a more reliable way
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
           
            try:
                #Generate speech with slower speed for better clarity
                tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
                tts.save(temp_path)
               
                #Play the audio with improved method
                play_audio(temp_path)
            except Exception as e:
                print(f"Voice generation/playback error: {e}")
            finally:
                #Ensure file is deleted with retries
                max_retries = 3
                for _ in range(max_retries):
                    try:
                        os.unlink(temp_path)
                        break
                    except PermissionError:
                        time.sleep(0.1)  #Wait and retry
                    except Exception as e:
                        print(f"Error deleting temp file: {e}")
                        break
        except Exception as e:
            print(f"Voice worker error: {e}")
        finally:
            is_speaking = False
            voice_queue.task_done()


#Start voice thread with higher priority
voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()


#Load optimized YOLO model
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit(1)


#Priority objects with enhanced warnings and synonyms
IMPORTANT_OBJECTS = {
    'person':    {'name': 'Person', 'priority': 1, 'warning': 'Person approaching!', 'caution': 'Person nearby'},
    'door':      {'name': 'Door', 'priority': 1, 'warning': 'Door close ahead!', 'caution': 'Door to your'},
    'stairs':    {'name': 'Stairs', 'priority': 1, 'warning': 'Stairs warning!', 'caution': 'Stairs approaching'},
    'car':       {'name': 'Car', 'priority': 2, 'warning': 'Vehicle approaching!', 'caution': 'Vehicle nearby'},
    'dog':       {'name': 'Dog', 'priority': 2, 'warning': 'Dog close by!', 'caution': 'Animal nearby'},
    'chair':     {'name': 'Chair', 'priority': 2, 'warning': 'Chair directly ahead!', 'caution': 'Chair to your'},
    'cell phone': {'name': 'Mobile phone', 'priority': 2, 'warning': 'Phone close ahead!', 'caution': 'Phone nearby'},
    'laptop':     {'name': 'Laptop', 'priority': 2, 'warning': 'Laptop ahead!', 'caution': 'Laptop near you'},
    'cup':        {'name': 'Cup', 'priority': 2, 'warning': 'Cup ahead!', 'caution': 'Cup to your'},
    'bed':        {'name': 'Bed', 'priority': 2, 'warning': 'Bed in front of you!', 'caution': 'Bed nearby'},
    'pizza':      {'name': 'Food', 'priority': 2, 'warning': 'Food detected nearby!', 'caution': 'Food close to you'},
    'sandwich':   {'name': 'Food', 'priority': 2, 'warning': 'Sandwich in front of you!', 'caution': 'Sandwich nearby'},
    'sneakers':   {'name': 'Sneakers', 'priority': 2, 'warning': 'Sneakers in the way!', 'caution': 'Shoes near your path'},
    'shoe':       {'name': 'Sneakers', 'priority': 2, 'warning': 'Shoes ahead!', 'caution': 'Sneakers nearby'}
}


#Add synonyms for better detection
OBJECT_SYNONYMS = {
    'person': ['human', 'man', 'woman', 'people'],
    'door': ['gate', 'entrance'],
    'car': ['vehicle', 'truck', 'bus'],
    'dog': ['cat', 'pet', 'animal'],
    'chair': ['seat', 'stool'],
    'cell phone': ['phone', 'mobile', 'smartphone'],
    'laptop': ['computer', 'notebook'],
    'cup': ['glass', 'mug'],
    'bed': ['cot', 'mattress']
}


#Camera setup
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
except Exception as e:
    print(f"Camera error: {e}")
    exit(1)


#Timing control
TARGET_FPS = 1.5
last_processing_time = time.time()
last_announce_time = 0
last_warning_time = 0


#Initial voice test with confirmation
voice_queue.put("Vision assistance system activated and ready")


def make_announcement(label, relative_size, position_x, frame_width):
    #Check synonyms first
    for main_label, synonyms in OBJECT_SYNONYMS.items():
        if label in synonyms:
            label = main_label
            break
   
    obj_info = IMPORTANT_OBJECTS.get(label)
    if not obj_info:
        return None
   
    position = "left" if position_x < frame_width/3 else "right" if position_x > 2*frame_width/3 else "front"
   
    if relative_size > 0.4:  #Very close
        return f"Warning! {obj_info['warning']}"
    elif relative_size > 0.25:  #Close
        if position in ['left', 'right']:
            return f"Caution! {obj_info['caution']} {position}"
        return f"Caution! {obj_info['caution']}"
    elif relative_size > 0.15:  #Nearby
        return f"{obj_info['name']} detected {position}"
    return None


def process_detection(label, score, box, frame, small_frame):
    if label in IMPORTANT_OBJECTS or any(label in synonyms for synonyms in OBJECT_SYNONYMS.values()):
        #Scale coordinates
        x1, y1, x2, y2 = box
        x1 = int(x1 * frame.shape[1] / small_frame.shape[1])
        y1 = int(y1 * frame.shape[0] / small_frame.shape[0])
        x2 = int(x2 * frame.shape[1] / small_frame.shape[1])
        y2 = int(y2 * frame.shape[0] / small_frame.shape[0])
       
        #Calculate metrics
        relative_size = ((x2 - x1) * (y2 - y1)) / (frame.shape[1] * frame.shape[0])
        center_x = (x1 + x2) / 2
       
        #Update object counter
        object_counter[label] = object_counter.get(label, 0) + 1
       
        #Visual feedback
        color = (0, 0, 255) if relative_size > 0.3 else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                  f"{IMPORTANT_OBJECTS.get(label, {}).get('name', label)} {relative_size:.1f}",
                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                  0.6, color, 2)
       
        return {
            'label': label,
            'size': relative_size,
            'priority': IMPORTANT_OBJECTS.get(label, {}).get('priority', 3),
            'position': center_x
        }
    return None


while cap.isOpened():
    #Control frame rate
    current_time = time.time()
    if current_time - last_processing_time < 1/TARGET_FPS:
        time.sleep(0.01)  #Small sleep to prevent busy waiting
        continue
    last_processing_time = current_time
   
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        time.sleep(0.1)
        continue


    #Run detection
    small_frame = cv2.resize(frame, (320, 240))
   
    try:
        results = yolo_model(small_frame, verbose=False, imgsz=320)
    except Exception as e:
        print(f"Detection error: {e}")
        continue


    current_detections = []
    object_counter = {}  #Reset counter for this frame
   
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()


        for i in range(min(5, len(boxes))):  #Process up to 5 objects now
            if scores[i] > 0.3:  #Confidence threshold
                label = yolo_model.names[int(labels[i])]
                detection = process_detection(label, scores[i], boxes[i], frame, small_frame)
                if detection:
                    current_detections.append(detection)


    #Voice announcements with better prioritization
    if current_detections:
        #Sort by priority then size (descending)
        current_detections.sort(key=lambda x: (-x['priority'], -x['size']))
       
        #Get the most important detection
        best_detection = current_detections[0]
       
        announcement = make_announcement(
            best_detection['label'],
            best_detection['size'],
            best_detection['position'],
            frame.shape[1]
        )
       
        #Immediate warnings for close objects
        if best_detection['size'] > 0.3 and (current_time - last_warning_time > 1):
            if announcement and not is_speaking and voice_queue.empty():
                voice_queue.put(announcement)
                last_warning_time = current_time
        #Regular announcements
        elif current_time - last_announce_time > 3:
            if announcement and not is_speaking and voice_queue.empty():
                voice_queue.put(announcement)
                last_announce_time = current_time


    #Display object count and status
    fps = 1/(time.time() - last_processing_time)
    status_text = f"FPS: {fps:.1f} | Objects: {sum(object_counter.values())}"
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
   
    #Display object counts per type
    for i, (obj, count) in enumerate(object_counter.items()):
        cv2.putText(frame, f"{obj}: {count}", (10, 60 + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
   
    cv2.imshow("Vision Assist", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



