# sailboat_CV

## extract frames from youtube videos

`python utils/youtube_utils/frame_extract.py --file utils/youtube_utils/urls.txt --quality 4k --cookies utils/youtube_utils/youtube_cookies.txt`

## filter them to keep only photos where the model detects pennons

`python utils/filter_youtube_photos_by_yolo.py`

## launch label studio 

`cd label-studio`
`docker-compose up -d`
