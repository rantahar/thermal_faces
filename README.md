# Thermal face recognition

An machine learning model for detecting faces in thermal camera videos.

The model used here is very small, since our data is small and relatively repetitive. Training a new
model for any new purpose is recommended.

## The Model

We use a similar approach to, for example, https://github.com/peiyunh/tiny, but simplify it considerably.
We train a convolutional model to detect faces in 32x32 pixels of temperature values. If a full detection
the image is partitioned into different sized sections and the model is applied to each section. For multiple
overlapping detections, we keep the section with largest confidence score produced by the detection model.

The detection model is implemented in thermal_face_detector/reduce_model.py.

## Usage

### Training

Training data should be preprocessed using the process_data script. For this, the temperature frames should
first be saved as individual numpy files (using numpy.save).

Use the label_images script from https://github.com/rantahar/seq_file to generate the labeled data.

For each numpy file, there should be a json file
with same base name should that contains labels. For example, the temperature data could be in a file names
`frame_100.npy` and the json file `frame_100.json` should contain labels in the following format:

```
{
    "labels": [
        {
            "x": 70,
            "y": 285,
            "l": 4
        },
        {
            "x": 186,
            "y": 253,
            "l": 1
        }
    ]
}
```

Here x and y are the coordinates and l is the label. Label 1 stands for forehead and 4 stands for nose. We 
currently only use these two labels.

To process this data into individual 32x32 training images, run for example

```bash
python ./process_data.py --data_path=data/
```

Next, train the model using 

```bash
python ./train_subsection_model.py --units=16 --negatives=5 --region_size=32 --data_path=data --save_path=saved --num_epochs=10000
```

### Processing a video

Now that you have a trained model file, you can apply it to a video. To do this, run for example

```bash
python process_video.py --model_file saved/10000 --video_file videos/video_1.seq  --new_region_threshold=10 --update_threshold=5
```


Before processing an entire video, you might prefer to test your model on a single frame. For this you can run

```bash
python apply_to_subframe.py --image_file data/frame_1.npy --model_file saved/10000 --step_fraction=0.1 --threshold=10
```

This will display the image with labeled regions.

## How it works (aka details)

### Data extraction

The face detection pipeline relies on a model that detects the existance of a face in an image. To train this model,
we must extract positive and negative examples from the training data. Positive samples are based on the manually
generated labels provided by the user. For each labeled nose, we extract positive example region a forehead is also
labeled reasonable close above the nose label. The example image is rescaled to the desired training size.

Negative examples are randomly selected from each frame. They contain all possible region sizes and are rescaled to
the desired size. Since the video frames will contain many more negative than positve examples, for efficiency of
training, we use several negative examples for each positive example. However, the ratio of positive to negative
cannot reasonably match the actual ratio.






