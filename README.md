# sailboat_CV

## Dataset 

### Download
`python utils/youtube/download_videos.py --quality 1080p --file utils/youtube/urls.txt`

### Cut videos



### Process dataset
`python utils/dataset/process_raw_dataset.py path_to_dataset --fps 25 --resolution 1080`

### Analyze raw dataset 

`python utils/dataset/analyze_raw_dataset.py path_to_dataset`

### Filter with model

`python utils/dataset/filter_photos_with_model.py path_to_folder --confidence 0.1`

### Harmonize source distribution

`python utils/dataset/harmonize_source_distribution.py flattened_images/ --output-dir final_images --max-per-source 30`

## launch label studio 

`cd label-studio`
`docker-compose up -d`
