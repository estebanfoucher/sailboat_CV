# sailboat_CV

# Raw data processing 

## Download

youtube:

`python utils/youtube/download_videos.py --file utils/youtube/urls.txt`

local:
put all local files in your well named folder.

## Cut videos
(optional)
`python utils/video/trim_videos_to_duration.py path_to_video_folder --max-duration 120 --output-dir trimmed_videos`

## Analyze videos 

`python utils/dataset/video_folder_report.py youtube_downloads/videos`

## extract frames 

`python utils/video/extract_frames.py --directory downloaded_data/videos --step 15`

## Process dataset
reorganize dataset as such
`python utils/dataset/raw_dataset/process_raw_dataset.py path_to_dataset`

## Analyze raw dataset 
then analyze it:
`python utils/dataset/analyze_raw_dataset.py path_to_dataset`

## Extract frames from videos

`python utils/video/extract_frames.py --directory tell_tales_seb_01_07_2025/videos/ --step 10`

## Filter with model

`python utils/dataset/filter_photos_with_model.py path_to_photo_folder --confidence 0.1`

## Unnest folder

`python utils/dataset/unnest.py --input tell_tales_seb_01_07_2025 --output flattened_images`

## Harmonize source distribution

analyze source repartition :
`python utils/dataset/image_source_report.py flattened_images`

put a limit to image per source to avoid overrepresentation :
`python utils/dataset/harmonize_source_distribution.py flattened_images/ --output-dir final_images --max-per-source 30`

# Sync to AWS

`aws s3 sync processed_images s3://bucket-sail-s3/frames/`

# Launch label studio 

`cd label-studio`
`docker-compose up -d`

## Export and prepare dataset

## Remove prefixes

`python utils/label_studio/remove_label_prefix.py label_folder`

## Download images from labels

`python utils/download_images_from_s3_and_label_json.py labels/labels-04.json data/datasets/labels-04/images`

## Split dataset 

`python utils/dataset/split_dataset.py data/datasets/labels-04`

## Oversample train split 

random picking images in source dataset and adding it if it makes you closer to the target class distribution (up to a number of iterations)

`python utils/augmentation/oversample.py data/splitted_datasets/labels-04/train`

## Analyze label distribution in a dataset 

`python utils/augmentation/print_distribution.py data/splitted_datasets/labels-04/train`

# Training

execute notebooks on colab

# Integrate new model in label studio ML backend

cp paste model in model folder in yolo_server instance and relaunch it

# Track test

need GPU

change video source in file : `python utils/model/track_ultralytics.py` 

