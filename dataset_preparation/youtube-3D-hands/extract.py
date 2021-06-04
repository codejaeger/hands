import json

dataset_path = "../storage/YouTube-3D-Hands/youtube_test.json"
video = "FtQ8o88Jq0s"

with open (dataset_path, 'r') as f:
    d = json.load(f)

images = []
annotations = []

for i in d['images']:
    if i['name'].split('/')[1] == video:
        images.append(i)

id_list = [d['images'][i]['id'] for i in range(len(images))]

for i in d['annotations']:
    if i['image_id'] in id_list:
        annotations.append(i)

dt = {}

dt['images'] = images
dt['annotaions'] = annotations
with open('400_youtube_test.json', 'w') as f:
    json.dump(dt, f)