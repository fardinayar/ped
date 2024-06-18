import pickle
import json
import numpy as np

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou

def update_pickle_with_looking_scores(pickle_data, json_data):
    for vid_id, vid_info in pickle_data.items():
        for ped_id, ped_info in vid_info['ped_annotations'].items():
            for frame_index, frame in enumerate(ped_info['frames']):
                frame_key = f"{vid_id}/{str(frame).rjust(5, '0')}.png"
                assert frame_key in json_data, f'missing {frame_key}'
                json_boxes = json_data[frame_key]['boxes']
                ped_box = ped_info['bbox'][frame_index]
                
                # Find the best matching box by IoU
                best_iou = 0
                best_score = 0
                for box in json_boxes:
                    iou = calculate_iou(ped_box[:4], box[:4])  # Compare only the coordinates, not the score
                    if iou > best_iou:
                        best_iou = iou
                        best_score = box[4]
                
                # Assuming there's a field to store the looking score in the pickle data structure
                if 'looking_score' not in ped_info:
                    ped_info['looking_score'] = []
                ped_info['looking_score'].append(best_score)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Example usage
pickle_file_path = '/home/jvn-server/Desktop/zahraa/exp1/Pedestrian_Crossing_Intention_Prediction/JAAD/data_cache/jaad_database.pkl'
json_file_path = '/home/jvn-server/Desktop/zahraa/exp2/Pedestrian_Crossing_Intention_Prediction/tools/looking/data.json'

pickle_data = load_pickle(pickle_file_path)
json_data = load_json(json_file_path)

update_pickle_with_looking_scores(pickle_data, json_data)
save_pickle(pickle_data, 'updated_pickle_file.pkl')