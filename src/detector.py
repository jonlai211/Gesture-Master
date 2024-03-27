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
                 min_tracking_confidence=0.5, max_records=10):
        self.results = None
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.project_root_dir = os.path.join(os.path.dirname(__file__), '..')
        self.record_count = {i: 0 for i in range(10)}
        self.max_records = max_records

        self.model = load_model(os.path.join(self.project_root_dir, "models", "ddnet_model.h5"))

    def detect(self, image, number, mode):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)

        landmark_list = []
        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks,
                                                  self.results.multi_handedness):
                # calculate bounding rect
                bounding_rect = calc_bounding_rect(image, hand_landmarks)
                # calculate landmark
                landmark_list = calc_landmark_list(image, hand_landmarks)

                processed_data = self.prepare_data(landmark_list)
                predicted_class, confidence = self.predict(processed_data)
                gesture_name = self.get_gesture_name(predicted_class)

                # draw landmark
                image = draw_bounding_rect(bounding_rect, image, bounding_rect)
                image = draw_landmarks(image, landmark_list)
                image = draw_info_text(
                    image,
                    bounding_rect,
                    handedness,
                    f"{gesture_name}, {confidence:.2f}"
                )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, landmark_list

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
            print(f"Expected 210 features, but got {len(jcd_features)}")
            return None
        model_input = jcd_features.reshape(1, -1)
        return model_input

    def predict(self, processed_data):
        if processed_data is None:
            print("No processed data to predict.")
            return None

        predictions = self.model.predict(processed_data)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        print(f"Predicted class: {predicted_class} with confidence: {confidence}")

        return predicted_class, confidence

    def get_gesture_name(self, predicted_class):
        gesture_names = {0: 'Open', 1: 'Index thumb', 2: 'Index middle', 3: 'Index middle', 4: 'Rock'}
        return gesture_names.get(predicted_class, "No zhi")

