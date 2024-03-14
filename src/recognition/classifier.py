import numpy as np


class GestureClassifier:
    def __init__(self, keypoint_classifier_model):
        self.keypoint_classifier = keypoint_classifier_model

    def classify(self, landmarks, point_history):
        # 对关键点进行分类
        keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
        hand_sign_id = self.keypoint_classifier.predict([keypoints])[0]

        return hand_sign_id
