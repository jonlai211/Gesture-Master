import csv
import os


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        current_script_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
        csv_path = os.path.join(dir_path, 'data', 'raw', f'keypoint_{number}.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return
