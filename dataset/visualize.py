import json
import numpy as np
import cv2
import os

def load_dataset(fp_data):
    with open(fp_data, "r") as file:
        data = json.load(file)

    return data

def retrieve_sample(data, image_name):
    annotation_list = []
    image = None
    for im in data['images']:
        if im['name'] == image_name:
            image = im
    
    for ann in data['annotations']:
        if ann['image_id'] == image['id']:
            annotation_list.append(ann)

    return annotation_list, image

def viz_2d(data, image_path, db_root):
    import imageio
    import matplotlib.pyplot as plt

    annotation_list, img = retrieve_sample(data, image_path)

    image = cv2.imread(os.path.join(db_root, 'data/', img['name']))
    image2 = cv2.resize(image, (img['width'], img['height']))
    print(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image2)
    
    for ann in annotation_list:
        vertices = np.array(ann['vertices'])
        plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='green', markersize=1)
    plt.show()
    
def viz_3d(data, image_path, db_root):
    import open3d as o3d
    from mano import Mano
    
    mano = Mano(f'{db_root}/MANO_RIGHT.pkl')

    annotation_list, _ = retrieve_sample(data, image_path)
    vertices = np.array(annotation_list[0]['vertices']) #Hardcoded the index
    
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(mano.triangles))
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_root = f'{project_root}/storage/'
    data1 = load_dataset(f'{project_root}/storage/correct_sample.json')
    # viz_2d(data1, data1['images'][0]['name'], db_root)
    # viz_3d(data1, data1['images'][0]['name'], db_root)
    
    data2 = load_dataset(f'{project_root}/storage/incorrect_sample.json')
    # viz_2d(data2, data2['images'][0]['name'], db_root)
    # viz_3d(data2, data2['images'][0]['name'], db_root)
    
    val_data = load_dataset(f'{project_root}/storage/YouTube-3D-Hands/youtube_train.json')
    viz_2d(val_data, val_data['images'][0]['name'], db_root)
    print(val_data['images'][0])
    # viz_3d(val_data, 'youtube/Raa0vBXA8OQ/video/frames/2298.png', db_root)
    # viz_3d(val_data, val_data['images'][0]['name'], db_root)
    
    print("Data keys:", [k for k in val_data.keys()])
    print("Image keys:", [k for k in val_data['images'][0].keys()])
    print("Annotations keys:", [k for k in val_data['annotations'][0].keys()])

    print("The number of images:", len(val_data['images']))
    print("The number of annotations:", len(val_data['annotations']))