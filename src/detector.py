import cv2
import os
import time
import logging
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import pdist, squareform
from utils.ui_tools import calc_bounding_rect, calc_landmark_list, draw_landmarks, draw_bounding_rect, draw_info_text
from utils.csv_tools import logging_csv
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Detector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, max_records=10, mouse_control_interval=8):
        self.results = None
        self.prev_pos = 0, 0
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.project_root_dir = os.path.join(os.path.dirname(__file__), '..')
        self.record_count = {i: 0 for i in range(20)}
        self.max_records = max_records
        self.mouse_control_interval = mouse_control_interval
        self.mouse_control_counter = 0

        self.model = load_model(os.path.join(self.project_root_dir, "models", "ddnet_model.h5"))
        self.last_gesture_name = None
        self.last_confidence = 0
        self.detect_counter = 0
        self.detect_interval = 10

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        # landmark_list = []
        gesture_name = None
        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks,
                                                  self.results.multi_handedness):
                bounding_rect = calc_bounding_rect(image_rgb, hand_landmarks)
                landmark_list = calc_landmark_list(image_rgb, hand_landmarks)

                self.detect_counter += 1
                if self.detect_counter >= self.detect_interval:
                    self.detect_counter = 0
                    processed_data = self.prepare_data(landmark_list)
                    predicted_class, confidence = self.predict(processed_data)
                    gesture_name = self.get_gesture_name(predicted_class)
                    self.last_gesture_name = gesture_name
                    self.last_confidence = confidence
                else:
                    gesture_name = self.last_gesture_name
                    confidence = self.last_confidence

                image_rgb = self.draw_image(image_rgb, bounding_rect, landmark_list, handedness, gesture_name,
                                            confidence)
        else:
            pass
            # logging.info("No hand detected.")

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image, gesture_name

    def execute_gesture_detection(self, image):
        processed_data = self.prepare_data(self.get_landmarks(image))
        predicted_class, confidence = self.predict(processed_data)
        return predicted_class, confidence

    def find_id_position(self, image, id, position):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        if self.results.multi_hand_landmarks is not None:
            myHand = self.results.multi_hand_landmarks[0]
            if id < len(myHand.landmark):
                lm = myHand.landmark[id]
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                position = (cx, cy)
        else:
            logging.info(f"No ID {str(id)} found.")
        #     TODO: fix losing ID leading to vibration
        return position

    def get_all_position(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

        return self.lmList, bbox

    def fingers_up(self, lmList):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers


    def get_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        landmark_list = []
        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks,
                                                  self.results.multi_handedness):
                landmark_list = calc_landmark_list(image_rgb, hand_landmarks)
        return landmark_list

    def draw_image(self, image, bounding_rect, landmark_list, handedness, gesture_name, confidence):
        image = draw_bounding_rect(bounding_rect, image, bounding_rect)
        image = draw_landmarks(image, landmark_list)
        image = draw_info_text(
            image,
            bounding_rect,
            handedness,
            f"{gesture_name}, {confidence:.2f}"
        )
        return image

    def record_data(self, key, number, mode, landmark_list, image):
        if key == 32 and mode == 1 and self.record_count[number] < self.max_records:
            logging_csv(number, mode, landmark_list)
            logging.info(f"Data recorded for gesture {number}: {self.record_count[number] + 1}/{self.max_records}")
            self.record_count[number] += 1

            timestamp = int(time.time())
            save_path = os.path.join(self.project_root_dir, "data", "images", str(number), f"{timestamp}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

            logging.info(f"Image saved to {save_path}")

            if self.record_count[number] == self.max_records:
                logging.info(f"Maximum records reached for gesture {number}.")

    def input_cal_jcd(self, landmark_list):
        landmarks_array = np.array(landmark_list)
        distances = pdist(landmarks_array, 'euclidean')
        jcd_features = squareform(distances)[np.tril_indices(len(landmarks_array), k=-1)]
        return jcd_features

    def prepare_data(self, landmark_list):
        if not landmark_list:
            return None

        jcd_features = self.input_cal_jcd(landmark_list)
        if len(jcd_features) != 210:
            logging.warning(f"Expected 210 features, but got {len(jcd_features)}")
            return None
        model_input = jcd_features.reshape(1, -1)
        return model_input

    def predict(self, processed_data):
        if processed_data is None:
            return None

        predictions = self.model.predict(processed_data, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        return predicted_class, confidence

    def get_gesture_name(self, predicted_class):
        gesture_names = {0: 'Open', 1: 'Fist', 2: 'Point', 3: 'Victory', 4: 'Three', 5: 'Four', 6: 'Shoot', 7: 'Likes',
                         8: 'Pinch', 9: 'Glory', 10: 'Rock', 11: 'Fxxk', 12: 'Spider', 13: 'LookDown',
                         14: 'OK', 15: 'Six'}
        return gesture_names.get(predicted_class, "No zhi")
