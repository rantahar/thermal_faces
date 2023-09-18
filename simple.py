import os
import numpy as np
import json

data_path = "data/train_data"

def load_npy_files(folder_path, validation_fraction=0.1):
    data = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            words = file_name.split('_')
            frame_index = int(words[-1].replace(".npy",""))
            video_name = "_".join(words[:-1])

            file_path = os.path.join(folder_path, file_name)
            array = np.load(file_path)

            json_file_path = os.path.join(folder_path, file_name.replace(".npy",".json"))
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            
            labeled_array = np.zeros_like(array)
            for label in json_data['labels']:
                x = label['x']
                y = label['y']
                l = label['l']
                if l == 1:
                    labeled_array[y, x] = 1

            if video_name in data:
                data[video_name].append((frame_index, array, labeled_array))
            else:
                data[video_name] = [(frame_index, array, labeled_array)]
    
    for key in data.keys():
        data[key].sort(key=lambda x: x[0])
    
    train_data = {}
    val_data = {}

    for video_name, frames in data.items():
        if len(frames) > 2:
            num_frames = len(frames)
            start_frame = int(np.floor((0.5-0.5*validation_fraction)*num_frames))
            end_frame = int(np.ceil((0.5+0.5*validation_fraction)*num_frames))

            val_frames = frames[start_frame:end_frame]
            val_data[video_name] = (
                np.stack([frame[1] for frame in val_frames], axis=0),
                np.stack([frame[2] for frame in val_frames], axis=0)
            )

            train_frames = frames[:start_frame] + frames[end_frame:]
            train_data[video_name] = (
                np.stack([frame[1] for frame in train_frames], axis=0),
                np.stack([frame[2] for frame in train_frames], axis=0)
            )

    return train_data, val_data


train_data, val_data = load_npy_files(data_path)

print()
for video in train_data:
    print(video, train_data[video][0].shape, train_data[video][1].shape)
print()
for video in val_data:
    print(video, val_data[video][0].shape, val_data[video][1].shape)

