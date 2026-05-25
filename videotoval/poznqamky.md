file structure:
├── model.py
├── poznqamky.md
├── test.py
├── training_data_load.py
├── train.py
└── video_processor.py

Title and brief description: snn to convert video to arousal/valence scores


Input: video

/srv/tribe-share/enma/saved_mri/Youtube_0-MYK_i6gl8_chunk_000/source_video.mp4


output: v/a scores

training data locations:

/srv/tribe-share/enma/saved_mri/Youtube_0-MYK_i6gl8_chunk_000/roi_timeseries.parquet

example:
frame_idx	time_sec	vmPFC	insula	ACC	valence	arousal
0	0.000000	0.141491	0.068780	0.150405	0.492120	-0.779778


done stuff:
video loader
video to image to spikes processor

stuff to make: 
-load valence/arousal scoresfor training
-correct dataloaders so it loads also the val/ar scores next to the video frames
-lower the fps from 30fps to 10fps
-