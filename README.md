# sailboat_CV

# Raw data processing 

## Download

youtube:
`python utils/youtube/download_videos.py --file utils/youtube/urls.txt --output downloads/videos/ --quality 1080p`

## Process dataset
reorganize dataset as such
`python utils/dataset/raw_dataset/process_raw_dataset.py downloads/`
or put manually in final dataset structure

now we have
downloads/
    - images/
    - videos/

## Analyze videos 

`python utils/dataset/video_folder_report.py downloads/videos`

## extract frames
choose step
`python utils/video/extract_frames.py --directory downloads/videos --step 15`

## filter images by time (optional)
`python utils/dataset/clean_images_by_intervals.py --txt_file utils/youtube/urls.txt --folder downloads/video_frames_extracted/videos/`

## Unnest folder
`python utils/dataset/unnest.py --input downloads/video_frames_extracted/ --output flattened_images_folder`

## Image only : add frame_ prefix
`python utils/images/add_frame_prefix.py downloads/images`

`cp -r -v downloads/images flattened_images_folder`

## Filter with model
`python utils/dataset/filter_photos_with_model.py flattened_images_folder --confidence 0.1`

## Harmonize source distribution

analyze source repartition :
`python utils/dataset/image_source_report.py flattened_images_folder`

put a limit to image per source to avoid overrepresentation :
`python utils/dataset/harmonize_source_distribution.py flattened_images_folder/ --output-dir final_images --max-per-source 30`

# Sync to AWS

`aws s3 sync processed_images s3://bucket-sail-s3/frames/`

# Launch label studio 

`cd label-studio`
`docker-compose up -d`

## Export and prepare dataset

## Remove prefixes

`python utils/label_studio/remove_label_prefix.py label_folder`

## Download images from labels

`python utils/download_images_from_s3_and_label_json.py labels/labels-05.json data/datasets/labels-05/images`

## Split dataset 

`python utils/dataset/split_dataset.py data/datasets/labels-05`

## Oversample train split 

random picking images in source dataset and adding it if it makes you closer to the target class distribution (up to a number of iterations)

`python utils/augmentation/oversample.py data/splitted_datasets/labels-05/train`

## Analyze label distribution in a dataset 

`python utils/augmentation/print_distribution.py data/splitted_datasets/labels-04/train`

# Training

execute notebooks on colab

# Integrate new model in label studio ML backend

cp paste model in model folder in yolo_server instance and relaunch it

# Track test

run test on custom video. configure your model and video input / output in config.yml
`python utils/track/track.py --config config.yml`

or batch infer on a folder containing multiple videos and images (not maintained) with 


