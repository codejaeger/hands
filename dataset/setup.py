import os
import pickle
import json
from create import extract_frames, create_dataset, save_dataset
from download import get_video_paths
import argparse
from video_lists import training_list, test_val_list

def add_metadata(vid_dict, images):
    for i in images:
        idt = i['name'].split('/')
        if idt[1] not in vid_dict:
            vid_dict[idt[1]] = []
        vid_dict[idt[1]].append(int(idt[4][:-4]))

def map_img_to_path(img_dct, images):
    for i in images:
        if i['id'] not in img_dct:
            img_dct[i['id']] = {}
        img_dct[i['id']]['path'] = i['name']

def map_img_to_ann(img_dct, annotations):
    for idx, ann in enumerate(annotations):
        if ann['image_id'] not in img_dct:
            img_dct[ann['image_id']] = {}
        if 'ann' not in img_dct[ann['image_id']]:
            img_dct[ann['image_id']]['ann'] = []
        img_dct[ann['image_id']]['ann'].append(idx)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Create dataset from YouTube video frames.')
    parser.add_argument('--db',
                        help='Video ID.',
                        required=False,
                        default=f'{project_root}/storage',
                        type=str)

    args = parser.parse_args()

    print('Load train dataset.')
    with open(f'{args.db}/YouTube-3D-Hands/youtube_train.json') as f:
        train_data = json.load(f)
    # print('Load test dataset.')
    # with open(f'{args.db}/YouTube-3D-Hands/youtube_test.json') as f:
    #     test_data = json.load(f)
    # print('Load val dataset.')
    # with open(f'{args.db}/YouTube-3D-Hands/youtube_val.json') as f:
    #     val_data = json.load(f)

    video_frames = {}
    # train_image_to_ann = {}
    # test_image_to_ann = {}
    val_image_to_ann = {}
    videos_out_path = []

    # add_metadata(video_frames, train_data['images'])
    # add_metadata(video_frames, test_data['images'])
    # add_metadata(video_frames, val_data['images'])

    # map_img_to_path(train_image_to_ann, train_data['images'])
    # map_img_to_path(test_image_to_ann, test_data['images'])
    # map_img_to_path(val_image_to_ann, val_data['images'])

    videos_out_path = get_video_paths(training_list, f'{args.db}/data/youtube')
    extract_frames(videos_out_path, video_frames)
    # videos_out_path = get_video_paths(test_val_list, f'{args.db}/data/youtube')
    # extract_frames(videos_out_path, video_frames)

    
    # map_img_to_ann(train_image_to_ann, train_data['annotations'])
    # map_img_to_ann(test_image_to_ann, test_data['annotations'])
    # map_img_to_ann(val_image_to_ann, val_data['annotations'])

    # X_val, Y_val, l_val = create_dataset(args.db, val_image_to_ann, val_data['annotations'])

    # save_dataset(X_val, Y_val, l_val, f'{args.db}/validation/')


    

    





    