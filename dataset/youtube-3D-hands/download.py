from pytube import YouTube

import json
import os
from os.path import join

from video_lists import training_list, test_val_list
YT_ROOT = 'https://youtu.be/'

def get_video_paths(videos_id_list, out_dir_root):
    """Download images for each video in the list.
    
    Args:
        videos_id_list: The list of YouTube video IDs.
        out_dir_root: Output directory.
    """
    videos_path_list = []
    for video_id in videos_id_list:
        try:
            print(video_id)
            out_path = download_video(video_id, out_dir_root)
            print(out_path)
            videos_path_list.append((out_path, video_id))
        except Exception as e:
            print(e)
    print(videos_path_list)
    return videos_path_list

def download_video(yt_id, out_dir_root):
    """Download a video.
    
    Args:
        yt_id: YouTube video ID.
        out_dir_root: Output directory.

    Returns:
        Filepath to the video directory.
    """
    out_path = join(out_dir_root, yt_id, 'video')
    os.makedirs(out_path, exist_ok=True)
    
    vid_path = join(out_path, 'raw.mp4')

    if not os.path.exists(vid_path):
        print('Download a video.')
        YouTube(join(YT_ROOT, yt_id)).streams \
            .filter(subtype='mp4', only_video=True) \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download(out_path)

        os.rename(join(out_path, os.listdir(out_path)[0]),
                  vid_path)
        
    return out_path

if __name__ == "__main__":
    # Take in storage path via sys argv
    get_video_paths(training_list, '../storage/data/')
    get_video_paths(test_val_list, '../storage/data/')