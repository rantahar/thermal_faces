import os
import numpy as np

data_path = "data/train_data"

def load_npy_files(folder_path, validation_fraction=0.1):
    data = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            words = file_name.split('_')
            frame_index = int(words[-1].replace(".npy",""))
            video_name = "_".join(words[:-1])

            file_path = os.path.join(folder_path, file_name)
            matrix = np.load(file_path)

            if video_name in data:
                data[video_name].append((frame_index, matrix))
            else:
                data[video_name] = [(frame_index, matrix)]
    
    for key in data.keys():
        data[key].sort(key=lambda x: x[0])
    
    train_data = {}
    val_data = {}

    for video_name, frames in data.items():
        if len(frames) > 2:
            num_frames = len(frames)
            start_frame = int(np.floor((0.5-0.5*validation_fraction)*num_frames))
            end_frame = int((np.ceil(0.5+0.5*validation_fraction)*num_frames))

            val_frames = frames[start_frame:end_frame]
            val_data[video_name] = np.stack([frame[1] for frame in val_frames], axis=0)

            train_frames = frames[:start_frame] + frames[end_frame:]
            train_data[video_name] = np.stack([frame[1] for frame in train_frames], axis=0)

    return train_data, val_data




