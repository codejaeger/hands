import json
import numpy as np
import os
import cv2
import glob
from os.path import join
import imageio

def bounding_box(points):
    x_coordinates, y_coordinates, _ = zip(*points)

    return np.array([[min(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]])

def extract_frames(videos_out_path, video_metadata, image_paths={}):

    for out_path, video_id in videos_out_path:
        vid_path = join(out_path, 'raw.mp4')
        f_out_path = join(out_path, 'frames')
        os.makedirs(f_out_path, exist_ok=True)
        print(vid_path)
        vidcap = cv2.VideoCapture(vid_path)
        count = 0
        list_iter = 0
        saved_frames = len(glob.glob(join(out_path, 'frames', '*.png')))
        
        if video_id not in video_metadata or saved_frames == len(video_metadata[video_id]): continue
        print(video_id)
        print('Extract frames.')

        while vidcap.isOpened():
            success, image = vidcap.read()
            if list_iter == len(video_metadata[video_id]): break
            if video_metadata[video_id][list_iter] == count:
                if success and list_iter >= saved_frames:
                    cv2.imwrite(join(f_out_path, '%d.png') % count, image)  
                elif not success:
                    print('Error creating frames.')
                    break
                list_iter += 1
            count += 1
        
        cv2.destroyAllWindows()
        vidcap.release()

def create_dataset(project_root, img_dct, annotations, pad=50):
    import matplotlib.pyplot as plt
    images = []
    vertices = []
    is_left = []
    for im in img_dct.keys():
        try:
            image = cv2.imread(join(project_root, 'data/'+img_dct[im]['path']))
        except:
            continue
        if image is None:
            continue
        for ann_idx in img_dct[im]['ann']:
            ann = annotations[ann_idx]
            #loss of precision
            bbx = bounding_box(ann['vertices']).astype(int)
            # print(bbx)
            cropped_img = image[bbx[0][1]-pad:bbx[1][1]+pad, bbx[0][0]-pad:bbx[1][0]+pad, :]
            cropped_vertices = np.array(ann['vertices'])
            cropped_vertices[:, 0] -= bbx[0][0]-pad
            cropped_vertices[:, 1] -= bbx[0][1]-pad
            images.append(cropped_img)
            vertices.append(cropped_vertices)
            is_left.append(ann['is_left'])
    return images, vertices, is_left

def save_dataset(X_val, Y_val, l_val, out_path):
    img_out_path = join(out_path, 'frames')
    os.makedirs(img_out_path, exist_ok=True)
    img_paths = []
    for idx, im in enumerate(X_val):
        img_paths.append(join(img_out_path, '%d.png') % idx)
        cv2.imwrite(join(img_out_path, '%d.png') % idx, im)
    
    dataset = {}
    dataset['X_val'] = img_paths
    dataset['Y_val'] = Y_val
    dataset['l_val'] = l_val

    json.dump(dataset, join(out_path, 'dataset.json'), indent = 4)
    