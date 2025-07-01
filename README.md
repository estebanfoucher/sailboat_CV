# sailboat_CV

## Dataset processing 

### Download

youtube:

`python utils/youtube/download_videos.py --quality 1080p --file utils/youtube/urls.txt`

local:
put all local files in your well named folder.

### Cut videos
(optional)
`python utils/video/trim_videos_to_duration.py path_to_video_folder --max-duration 120 --output-dir trimmed_videos`

### Process dataset
reorganize dataset as such
`python utils/dataset/process_raw_dataset.py path_to_dataset --skip_downsample True`

### Analyze raw dataset 
then analyze it:
`python utils/dataset/analyze_raw_dataset.py path_to_dataset`

### Extract frames from videos

`python utils/video/extract_frames.py --directory tell_tales_seb_01_07_2025/videos/ --step 10`

### Filter with model

`python utils/dataset/filter_photos_with_model.py path_to_photo_folder --confidence 0.1`

### Unnest folder

`python utils/dataset/unnest.py --input tell_tales_seb_01_07_2025 --output flattened_images`

### Harmonize source distribution

analyze source repartition :
`python utils/dataset/image_source_report.py flattened_images`

put a limit to image per source to avoid overrepresentation :
`python utils/dataset/harmonize_source_distribution.py flattened_images/ --output-dir final_images --max-per-source 30`

## Sync to AWS

`aws s3 sync processed_images s3://bucket-sail-s3/frames/`

## Launch label studio 

`cd label-studio`
`docker-compose up -d`
