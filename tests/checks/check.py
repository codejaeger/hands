import tensorflow as tf 
import numpy as np  
import imageio
import json
import plotly.express as px
import plotly.graph_objects as go
import cv2 as cv
import matplotlib.pyplot as plt
# from psbody.mesh import Mesh
import scipy
# %matplotlib widget
from PIL import Image
from mano import Mano
import matplotlib.pyplot as plt
from tqdm import tqdm
import tqdm.notebook as tq
from tqdm import tqdm

from utils import read_mano, scatter3d, update_fig_range, plot_2d_pts_on_img

def visualize_in_orig_image(image, annotations):
    """
    Visualize hand annotations and image on uncropped image
    """
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    vertices = np.array(annotations)
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='green', markersize=1)
    plt.show()

img_path = "../storage/sample_friehand/00000000.jpg"
I = imageio.imread(img_path)
# plt.imshow(I)

len(I[0])

with open('../storage/sample_freihand_1.json') as f:
    training = json.load(f)

# with open('../storage/training_K.json') as f:
#     training_K = json.load(f)    

# with open('../storage/training_mano.json') as f:
#     training_mano = json.load(f)

K, mano, xyz = training['K'], training['mano'], training['xyz']

from util.fh_utils import *
from util.model import *

mano = np.array(mano)
K = np.array(K)

poses, shapes, uv_root, scale = split_theta(mano)
focal, pp = get_focal_pp(K)
xyz_root = recover_root(uv_root, scale, focal, pp)

# set up the hand model and feed hand parameters
renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
msk_rendered, V1 = renderer.render(K, img_shape=I.shape[:2])

proj = projectPoints(V1, K)
print(proj.shape)        
s_x = np.std(proj[:, 0]) / np.std(V1[:, 0])
z_cam = s_x *(V1[:, 2]).reshape(-1, 1)
V_gt = np.concatenate((proj[:, :2], z_cam), axis=1)

proj_mat = K
proj_mat = np.hstack([proj_mat, np.zeros((3, 1))])
proj_mat = np.vstack([proj_mat, [0, 0, 0, 1]])
proj_mat[2, 2] = s_x
 
# visualize_in_orig_image(I, V_gt)

print(V_gt[:3, :])

# with open('../storage/sample_friehand/sample_friehand.json') as f:
#     annotation_json = json.load(f)

# V_gt = np.array(annotation_json['anns']).astype(np.float32)

# visualize_in_orig_image(I, V_gt)

# fig = plot_2d_pts_on_img(I, V2)
# fig.show()

min_coords, max_coords = np.amin(V_gt, axis=0), np.amax(V_gt, axis=0)
min_uv, max_uv = min_coords[:2].astype(np.int), max_coords[:2].astype(np.int)
I_crop = I[min_uv[1]: max_uv[1], min_uv[0]: max_uv[0]]  # u: cols, v: rows

mat_crop = np.identity(4, dtype=np.float32)
mat_crop[0, 3] = -min_uv[0]
mat_crop[1, 3] = -min_uv[1]
V_crop = V_gt @ mat_crop[:3, :3].T + mat_crop[:3, 3]

# visualize_in_orig_image(I_crop, V_crop)

out_img_size = 224

cropped_height, cropped_width, _ = I_crop.shape
resize_scale = min(out_img_size/cropped_width, out_img_size/cropped_height)
resized_width, resized_height = (int(cropped_width*resize_scale), int(cropped_height*resize_scale))
I_resize = cv.resize(I_crop, (resized_width, resized_height), interpolation=cv.INTER_LINEAR)

mat_resize = np.identity(4, dtype=np.float32)
mat_resize[0, 0] = resize_scale
mat_resize[1, 1] = resize_scale
V_resize = V_crop @ mat_resize[:3, :3].T + mat_resize[:3, 3]

# visualize_in_orig_image(I_resize, V_resize)

I_pad = np.zeros([out_img_size, out_img_size, 3], dtype=np.float32)
u_pad_start, v_pad_start = int((out_img_size - resized_width) / 2), int((out_img_size - resized_height) / 2)
u_pad_end, v_pad_end = u_pad_start + resized_width, v_pad_start + resized_height
I_pad[v_pad_start : v_pad_end, u_pad_start : u_pad_end] = I_resize

mat_pad = np.eye(4, dtype=np.float32)
mat_pad[0, 3] = u_pad_start
mat_pad[1, 3] = v_pad_start
V_pad = V_resize @ mat_pad[:3, :3].T + mat_pad[:3, 3]

# fig = plot_2d_pts_on_img(I_pad, V_pad)
# fig.show()

I_normalize = I_pad.copy()
I_normalize /= 255

shift_z = np.min(V_pad[:, 2])
scale_z = np.max(V_pad[:, 2])-np.min(V_pad[:, 2])

# V_normalize = V_pad.copy()
# V_normalize[:, :2] = V_normalize[:, :2] / ( out_img_size/2 ) - 1 
# V_normalize[:, 2] = (V_normalize[:, 2] - shift_z) / (scale_z/2 ) - 1 

mat_normalize = np.eye(4, dtype=np.float32)
mat_normalize[0, 0] = 2/out_img_size
mat_normalize[1, 1] = 2/out_img_size
mat_normalize[2, 2] = 2/scale_z

mat_normalize[0:3, 3] = [-1, -1, -(shift_z*2)/scale_z-1]
V_normalize = np.ones((778, 4))
V_normalize[:, :3] = V_pad.copy()
V_normalize = (mat_normalize @ V_normalize.T).T[:, :3]

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(I_normalize)

# ax2 = fig.add_subplot(122)
# ax2.scatter(V_normalize[:, 0], V_normalize[:, 1])
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(1, -1)
# ax2.set_aspect("equal")

print(np.sum(I_normalize), np.sum(V_normalize))

import cv2
CROP_OFFSET_RANGE = [-5, 5]
ROTATION_RANGE = [-180, 180]
random_angle_in_degrees = np.random.rand() * (ROTATION_RANGE[1]-ROTATION_RANGE[0]) - ROTATION_RANGE[0]
random_scale = np.random.rand()*0.1 + 1.0
cv_mat_rot_scale = cv2.getRotationMatrix2D((out_img_size/2, out_img_size/2), random_angle_in_degrees, random_scale)
rotated_and_scaled_image = cv2.warpAffine(I_normalize, cv_mat_rot_scale, (out_img_size, out_img_size), borderValue=0, flags=cv2.INTER_NEAREST)
rotated_and_scaled_image *= random_scale

# print(cv_mat_rot_scale)
mat_rot_scale = np.eye(4, dtype=np.float32)
mat_rot_scale[:2, :2] = cv_mat_rot_scale[:2, :2]
tmp = np.zeros((778, 4))
tmp[:,:3] = V_normalize
tmp[:, 3] = 1
rotated_scaled_vertices = tmp @ mat_rot_scale.T
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(rotated_and_scaled_image)

# ax2 = fig.add_subplot(122)
# ax2.scatter(rotated_scaled_vertices[:, 0], rotated_scaled_vertices[:, 1])
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(1, -1)
# ax2.set_aspect("equal")

imagenet_mean=np.array([0.485, 0.456, 0.406])
imagenet_std=np.array([0.229, 0.224, 0.225])
intensity_normalized_image = (rotated_and_scaled_image - imagenet_mean)/imagenet_std

print(np.min(V_normalize), np.max(V_normalize))
shift_z

V_normalize

# unnormalize
V_unnormalize = V_normalize.copy()
# V_unnormalize[:, :2] = (V_unnormalize[:, :2] + 1) * (out_img_size/2)
# V_unnormalize[:, 2] = (V_unnormalize[:, 2] + 1) * (scale_z/2) + shift_z

# unprocess
mat_process = mat_normalize @ mat_pad @ mat_resize @ mat_crop
mat_unprocess_3x3 = np.linalg.inv(mat_process[:3, :3])
mat_unprocess_t = mat_unprocess_3x3 @ (-mat_process[:3, 3])
V_unprocess = V_unnormalize @ mat_unprocess_3x3.T + mat_unprocess_t

visualize_in_orig_image(I, V_unprocess)

mat_process

V_orig = V_unprocess.copy()
V_orig[..., 2] = V_orig[..., 2] / proj_mat[2, 2]
V_orig[..., 0] = (V_orig[..., 0] - proj_mat[0, 2]) / proj_mat[0, 0] * V_orig[..., 2]
V_orig[..., 1] = (V_orig[..., 1] - proj_mat[1, 2]) / proj_mat[1, 1] * V_orig[..., 2]

print(V1[:3, :], V_orig[:3, :])

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(I)

# ax2 = fig.add_subplot(122)
# ax2.scatter(V_orig[:, 0], V_orig[:, 1])
# ax2.set_aspect("equal")

print(np.min(V_unprocess[:, 2]), np.max(V_unprocess[:, 2]))

n_verts_to_predict = 778

ref_i = rotated_and_scaled_image
ref_v = rotated_scaled_vertices
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(ref_i)

# ax2 = fig.add_subplot(122)
# ax2.scatter(ref_v[:, 0], ref_v[:, 1])
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(1, -1)
# ax2.set_aspect("equal")

# annotation = tf.constant([[[1, 1, 1], [2, 2, 2,]], [[1, 1, 1], [2, 2, 2,]]])
# annotation = tf.stack([annotation, tf.expand_dims(tf.ones(tf.shape(annotation)[:-1]), axis=-1)], axis=-1)
# annotation

ds_train1 = tf.data.Dataset.from_tensors(({"I": I_normalize, "proj_mat": proj_mat, "affine_mat": mat_process}, {"V": V_normalize[:n_verts_to_predict, :3], "annotations_xyz": V1}))
ds_train1 = ds_train1.batch(1)

ds_val1 = tf.data.Dataset.from_tensors(({"I": I_normalize, "proj_mat": proj_mat, "affine_mat": mat_process}, {"V": V_normalize[:n_verts_to_predict, :3], "annotations_xyz": V1}))
ds_val1 = ds_val1.batch(1)

print(ds_train1)

annotation_json_100 = [{'V': np.array(V1), 'K': K}]
annotation_json_100[0]['V']

# with open('../storage/training_.json') as f:
#     full_annotation = json.load(f)

with open('../storage/training_K.json') as f:
    full_annotation_K = json.load(f)

with open('../storage/training_mano.json') as f:
    full_annotation_mano = json.load(f)

len(full_annotation_mano)

annotation_json = []
for i in tqdm(range(len(full_annotation_mano))):
    mano = np.array(full_annotation_mano[i])
    K = np.array(full_annotation_K[i])

    poses, shapes, uv_root, scale = split_theta(mano)
    focal, pp = get_focal_pp(K)
    xyz_root = recover_root(uv_root, scale, focal, pp)
    
    # set up the hand model and feed hand parameters
    renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
    # print("Hellllllo")
    renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
    msk_rendered, V1 = renderer.render(K, img_shape=I.shape[:2])
    annotation_json.append({'V':V1, 'K':K})
with open('ann.json', 'w') as f:
    json.dump({'anns':annotation_json}, f)

# def process_data_wrapper(projection_func):
#     #Load these in process_ds for parallelisation
#     def f(image_path, annotation_xyz, *args):
#         image = np.array(Image.open(image_path.numpy()))
#         annotation_xyz = annotation_xyz.numpy()
#         image, annotation, proj_mat = projection_func(image, annotation_xyz, *args)
#         normalized_image, normalized_annotations, affine_mat = process_data(image, annotation)
    
#         return normalized_image, normalized_annotations, affine_mat, proj_mat
#     return f

# def youtube_projection(image, annotation, image_width, image_height, hand_is_left):
#     image_height = image_height.numpy()
#     image_width = image_width.numpy()
#     hand_is_left = hand_is_left.numpy()
#     ## Resize
#     image = Image.resize((image_width, image_height), 2)
#     ## Flip
#     flipped_image, flipped_vertices, flip_mat = flip_hand(image, annotation)
#     ## Return proj mat
#     return flipped_image, flipped_vertices, flip_mat

# def freihand_projection(image, annotation_xyz, projection_mat):
#     ## Apply projection mat
#     # print(annotation_xyz[:3, :])
#     annotation_uv = projectPoints(annotation_xyz, projection_mat)
#     projection_mat = np.hstack([projection_mat, np.zeros((3, 1))])
#     projection_mat = np.vstack([projection_mat, [0, 0, 0, 1]])
#     ## Scale z coord
#     s_x = np.std(annotation_uv[:, 0]) / np.std(annotation_xyz[:, 0])
#     z_cam = s_x * (annotation_xyz[:, 2]).reshape(-1, 1)
#     projection_mat[2, 2] = s_x
#     ## Add z scale to proj
#     annotation_uvd = np.concatenate((annotation_uv[:, :2], z_cam), axis=1)
#     ## Return modified proj mat
#     # print(annotation_uvd[:3, :])
#     return image, annotation_uvd, projection_mat

# def unproject_youtube(annotation, affine_mat, proj_mat):
#     mat_project = affine_mat @ proj_mat
#     mat_unproject = np.linalg.inv(mat_project[:3, :3])
#     inv_t = mat_unproject @ mat_project[:3, 3]
#     annotation = annotation @ mat_unproject.T + inv_t.T
#     return annotation

# def unproject_freihand(_annotation, _affine_mat, _proj_mat):
#     def f(annotation, affine_mat, proj_mat):
#         mat_project = affine_mat
#         mat_unproject = np.linalg.inv(mat_project[:3, :3])
#         inv_t = mat_unproject @ -mat_project[:3, 3]
#         annotation = annotation @ mat_unproject.T + inv_t.T
#         annotation[..., 2] = annotation[..., 2] / proj_mat[2, 2]
#         annotation[..., 0] = (annotation[..., 0] - proj_mat[0, 2]) / proj_mat[0, 0] * annotation[..., 2]
#         annotation[..., 1] = (annotation[..., 1] - proj_mat[1, 2]) / proj_mat[1, 1] * annotation[..., 2]
#         return annotation
#     _annotation = _annotation.numpy()
#     for idx, _ in enumerate(_annotation):
#         _annotation[idx] = f(_annotation[idx], _affine_mat[idx].numpy(), _proj_mat[idx].numpy())
#     # print(_annotation.shape)
#     return _annotation

# def process_data(I, V_gt):
#     min_coords, max_coords = np.amin(V_gt, axis=0), np.amax(V_gt, axis=0)
#     min_uv, max_uv = min_coords[:2].astype(np.int), max_coords[:2].astype(np.int)
#     I_crop = I[min_uv[1]: max_uv[1], min_uv[0]: max_uv[0]]  # u: cols, v: rows

#     mat_crop = np.identity(4, dtype=np.float32)
#     mat_crop[0, 3] = -min_uv[0]
#     mat_crop[1, 3] = -min_uv[1]
#     V_crop = V_gt @ mat_crop[:3, :3].T + mat_crop[:3, 3]
#     cropped_height, cropped_width, _ = I_crop.shape
#     resize_scale = min(out_img_size/cropped_width, out_img_size/cropped_height)
#     resized_width, resized_height = (int(cropped_width*resize_scale), int(cropped_height*resize_scale))
#     I_resize = cv.resize(I_crop, (resized_width, resized_height), interpolation=cv.INTER_LINEAR)

#     mat_resize = np.identity(4, dtype=np.float32)
#     mat_resize[0, 0] = resize_scale
#     mat_resize[1, 1] = resize_scale
#     V_resize = V_crop @ mat_resize[:3, :3].T + mat_resize[:3, 3]
    
#     I_pad = np.zeros([out_img_size, out_img_size, 3], dtype=np.float32)
#     u_pad_start, v_pad_start = int((out_img_size - resized_width) / 2), int((out_img_size - resized_height) / 2)
#     u_pad_end, v_pad_end = u_pad_start + resized_width, v_pad_start + resized_height
#     I_pad[v_pad_start : v_pad_end, u_pad_start : u_pad_end] = I_resize

#     mat_pad = np.eye(4, dtype=np.float32)
#     mat_pad[0, 3] = u_pad_start
#     mat_pad[1, 3] = v_pad_start
#     V_pad = V_resize @ mat_pad[:3, :3].T + mat_pad[:3, 3]

#     I_normalize = I_pad.copy()
#     I_normalize /= 255

#     shift_z = np.min(V_pad[:, 2])
#     scale_z = np.max(V_pad[:, 2])-np.min(V_pad[:, 2])
    
#     mat_normalize = np.eye(4, dtype=np.float32)
#     mat_normalize[0, 0] = 2/out_img_size
#     mat_normalize[1, 1] = 2/out_img_size
#     mat_normalize[2, 2] = 2/scale_z

#     mat_normalize[0:3, 3] = [-1, -1, -(shift_z*2)/scale_z-1]
    
#     V_normalize = np.ones((778, 4))
#     V_normalize[:, :3] = V_pad.copy()
#     V_normalize = (mat_normalize @ V_normalize.T).T[:, :3]
# #     print(np.sum(I_normalize), np.sum(V_normalize))
#     random_angle_in_degrees = np.random.rand() * (ROTATION_RANGE[1]-ROTATION_RANGE[0]) - ROTATION_RANGE[0]
#     random_scale = np.random.rand()*0.1 + 1.0
#     cv_mat_rot_scale = cv2.getRotationMatrix2D((out_img_size/2, out_img_size/2), random_angle_in_degrees, random_scale)
#     rotated_and_scaled_image = cv2.warpAffine(I_normalize, cv_mat_rot_scale, (out_img_size, out_img_size), borderValue=0, flags=cv2.INTER_NEAREST)
#     rotated_and_scaled_image *= random_scale

#     # print(cv_mat_rot_scale)
#     mat_rot_scale = np.eye(4, dtype=np.float64)
#     mat_rot_scale[:2, :2] = cv_mat_rot_scale[:2, :2]
#     tmp = np.zeros((778, 4))
#     tmp[:,:3] = V_normalize
#     tmp[:, 3] = 1
#     rotated_scaled_vertices = tmp @ mat_rot_scale.T
#     mat_process = mat_normalize @ mat_pad @ mat_resize @ mat_crop
#     # print(mat_process)
# #     print(rotated_and_scaled_image, rotated_scaled_vertices)
#     return I_normalize, V_normalize[:, :3], mat_process

# def get_raw_data_as_tf_dataset(storage_dir, images, annotations, dataset):
# #     image_dct = {}
# #     image_path = []
# #     image_width = []
# #     image_height = []
# #     hand_is_left = []
# #     annotation_idx = []
# #     map_img_to_path(image_dct, images)
# #     map_img_to_ann(image_dct, annotations)
# #     for idx, img in enumerate(image_dct.keys()):
# #         image_dct[img]['path'] = os.path.join(storage_dir, image_dct[img]['path'])
# #         for ann in image_dct[img]['ann']:
# #             image_path.append(image_dct[img]['path'])
# #             image_width.append(image_dct[img]['width'])
# #             image_height.append(image_dct[img]['height'])
# #             hand_is_left.append(annotations[ann]['is_left'])
# #             annotation_idx.append(ann)

# #         if idx >= 1:
# #             break
        
#     ## Friehand read
#     anns = []
#     image_path = []
#     proj_mat = []
#     for idx, ann in enumerate(annotations):
#         anns.append(ann['V'])
#         proj_mat.append(ann['K'])
#         image_path.append('../storage/sample_friehand/'+"00000000"[:(8-len(str(idx)))]+str(idx)+'.jpg')
#         # if idx==1:
#         #     break
#     ds_raw = tf.data.Dataset.from_tensor_slices((image_path, anns, proj_mat))
#     return ds_raw

# def process_ds(projection_func):
#     #Load image and annotations from text file later
#     def f(image_path, annotation, *args):
#         normalized_image, normalized_annotations, affine_mat, proj_mat = tf.py_function(
#             func = process_data_wrapper(projection_func),
#             inp = [image_path, annotation, *args],
#             Tout = [tf.float32, tf.float32, tf.float32, tf.float32]
#         )

#         annotation.set_shape([778, 3])
#         normalized_image.set_shape([out_img_size, out_img_size, 3])
#         normalized_annotations.set_shape([778, 3])
#         affine_mat.set_shape([4, 4])
#         proj_mat.set_shape([4, 4])

#         return (
#             {"I" : normalized_image, "proj_mat" : proj_mat, "affine_mat" : affine_mat},
#             {"V" : normalized_annotations, "annotations_xyz" : annotation}
#         )
#     return f
# #     return normalized_image, normalized_annotations

# def process_tf_dataset(ds_raw, dataset):
#     if dataset=="youtube":
#         ds_processed = ds_raw.map(
#                             lambda image_path, image_width, image_height, hand_is_left, annotation_id : process_ds(youtube_projection)(
#                                     image_path, annotation_id, image_width, image_height, hand_is_left))
#     else:
#         ds_processed = ds_raw.map(lambda image_path, annotation_id, proj_mat : process_ds(freihand_projection)(image_path, annotation_id, proj_mat))
#         ds_processed = ds_processed.apply(tf.data.experimental.ignore_errors())
#         ## Apply ignore errors
#     return ds_processed

# def get_processed_dataset_as_tf_dataset(storage_dir, images, annotations, dataset="freihand"):
#     ds_raw = get_raw_data_as_tf_dataset(storage_dir, images, annotations, dataset)
    
#     n_data = ds_raw.cardinality().numpy()
#     train_frac = 1
#     n_data_train = int(n_data * train_frac)
#     ds_train = ds_raw.take(n_data_train)
#     ds_val = ds_raw.skip(n_data_train)
#     ds_train = process_tf_dataset(ds_train, dataset)
# #     ds_train = ds_train.apply(tf.data.experimental.ignore_errors())
#     ds_val = process_tf_dataset(ds_val, dataset)

#     return ds_train, ds_val




# ds_train, ds_val = get_processed_dataset_as_tf_dataset(None, None, annotation_json_100)

# # ds_train = get_raw_data_as_tf_dataset(None, None, None)
# # ds_train = process_tf_dataset(ds_train, True)
# ds_train = ds_train.batch(1)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
# ds_val = ds_val.batch(1)
# print(ds_train, ds_val)

# for ds in ds_train:
#     fig = plt.figure()
#     ax1 = fig.add_subplot(121)
#     tf.debugging.check_numerics(ds[0]['I'][0], "asd")
#     ax1.imshow(ds[0]['I'][0].numpy())
#     # tf.print(ds[0]['I'][0])

#     ax2 = fig.add_subplot(122)
#     tf.debugging.check_numerics(ds[1]['V'][0], "asd")
#     vertex_set = ds[1]['V'][0].numpy()
#     ax2.scatter(vertex_set[:, 0], vertex_set[:, 1])
# #     ax2.set_xlim(-1, 1)
# #     ax2.set_ylim(1, -1)
#     tf.print(ds[1]['V'][0])
#     ax2.set_aspect("equal")
#     print(unproject_freihand(tf.convert_to_tensor([ds[1]['V'][0]]), tf.convert_to_tensor([ds[0]['affine_mat'][0]]), tf.convert_to_tensor([ds[0]['proj_mat'][0]])))
#     break
    
# #     break
    
# # for ds in ds_train1:
# #     fig = plt.figure()
# #     ax1 = fig.add_subplot(121)
# #     ax1.imshow(ds[0]['I'][0].numpy())
# # #     tf.print(ds[0]['I'][0])

# #     ax2 = fig.add_subplot(122)
# #     vertex_set = ds[1]['V'][0].numpy()
# #     ax2.scatter(vertex_set[:, 0], vertex_set[:, 1])
# #     ax2.set_xlim(-1, 1)
# #     ax2.set_ylim(1, -1)
# #     ax2.set_aspect("equal")
# #     print(np.min(ds[1]['V'][0].numpy()), np.max(ds[1]['V'][0].numpy()))



# def get_edge_mat(face_data, num_vert):
#     """
#     Get edge matrix of dimension Num_edges x Num_vertices
#     Example :- [[1, 0, 0, -1, 0...], .. .. ]
#     """
#     edge_list = []
#     for f in face_data:
#         edge_list.append((f[0], f[1]) if f[0]<f[1] else (f[1], f[0]))
#         edge_list.append((f[1], f[2]) if f[1]<f[2] else (f[2], f[1]))
#         edge_list.append((f[2], f[0]) if f[2]<f[0] else (f[0], f[2]))
#     edge_list = list(set(edge_list))
# #     print(edge_list)
#     edge_mat = np.zeros((len(edge_list), num_vert))
#     for idx, e in enumerate(edge_list):
#         edge_mat[idx, e[0]]=1
#         edge_mat[idx, e[1]]=-1
#     return edge_mat

# def get_sparse_edge_mat(edge_mat):
#     """
#     edge_mat: Num_edges_in_face*778 
#     """
#     edge_mat = scipy.sparse.csr_matrix(edge_mat)
#     edge_mat = edge_mat.tocoo()
#     indices = np.column_stack((edge_mat.row, edge_mat.col))
#     edge_mat = tf.SparseTensor(indices, edge_mat.data, edge_mat.shape)
#     edge_mat = tf.sparse.reorder(edge_mat)
#     return edge_mat

# mano = Mano('../storage/MANO_RIGHT.pkl')
# sparse_edge_mat = tf.convert_to_tensor(get_edge_mat(mano.triangles, len(mano.V_temp)), dtype=tf.float32)
# tensor_edge_mat = get_sparse_edge_mat(sparse_edge_mat)

# def loss_function(y_true, y_pred):
#     num_verts = tf.shape(y_true)[1]
#     num_edges = tf.shape(tensor_edge_mat)[0]
#     y_true = tf.transpose(y_true, perm=[1, 2, 0]) # 778 x 3 x N
#     y_true = tf.reshape(y_true, [num_verts, -1])
#     y_pred = tf.transpose(y_pred, perm=[1, 2, 0]) # 778 x 3 x N
#     y_pred = tf.reshape(y_pred, [num_verts, -1])
#     edge_true = tf.sparse.sparse_dense_matmul(tensor_edge_mat, y_true) # num_edges x 3N
#     edge_pred = tf.sparse.sparse_dense_matmul(tensor_edge_mat, y_pred) # num_edges x 3N
#     edge_pred = tf.reshape(edge_pred, [num_edges, 3, -1])
#     edge_true = tf.reshape(edge_true, [num_edges, 3, -1])
#     y_pred = tf.reshape(y_pred, [num_verts, 3, -1]) # 778 x 3 x N
#     y_pred = tf.transpose(y_pred, perm=[2, 0, 1]) # N x 778 x 3
#     y_true = tf.reshape(y_true, [num_verts, 3, -1])
#     y_true = tf.transpose(y_true, perm=[2, 0, 1]) # N x 778 x 3

# #     tf.print(tf.shape(y_true), tf.shape(tf.reduce_sum(tf.norm(y_true - y_pred, ord=1, axis=1), axis=0)), tf.shape(tf.reduce_sum(tf.abs(tf.norm(edge_true, ord='euclidean', axis=1)**2 - tf.norm(edge_pred, ord='euclidean', axis=1)**2), axis=0)), output_stream=sys.stdout)
# #     tf.print(tf.norm(edge_true, ord='euclidean', axis=1) - tf.norm(edge_pred, ord='euclidean', axis=1) , output_stream=sys.stdout)
# #     tf.print(tf.reduce_mean(tf.norm(y_true - y_pred, ord=1, axis=1), axis=-1) , output_stream=sys.stdout)
# #     tf.print(tf.reduce_mean(tf.reduce_mean(tf.abs(tf.norm(edge_true, ord=2, axis=1) - tf.norm(edge_pred, ord=2, axis=1)), axis=-1), axis=-1))
# #     tf.print(tf.reduce_max(tf.abs(tf.norm(edge_true, ord=2, axis=1) - tf.norm(edge_pred, ord=2, axis=1))), output_stream=sys.stdout)
# #     tf.autograph.trace()
# #     tf.autograph.trace()

#     return tf.reduce_mean(tf.reduce_sum(tf.norm(y_true - y_pred, ord=1, axis=2), axis=1)) + tf.reduce_mean(tf.reduce_sum(tf.abs(tf.norm(edge_true, ord='euclidean', axis=1)**2 - tf.norm(edge_pred, ord='euclidean', axis=1)**2), axis=0))
# #     return tf.reduce_mean(tf.reduce_sum(tf.norm(y_true - y_pred, ord=1, axis=2), axis=1))

# meta_dir = "../storage/metadata/"
# mano = Mano(meta_dir+'MANO_RIGHT.pkl')
# # v_temp_bbx = bounding_box(mano.V_temp, 0)
# # MANO_SCALE = tf.convert_to_tensor((v_temp_bbx[1,:]-v_temp_bbx[0,:])/2, dtype=tf.float32)
# # MANO_SCALE

# def get_edge_mat(face_data, num_vert):
#     """
#     Get edge matrix of dimension Num_edges x Num_vertices
#     Example :- [[1, 0, 0, -1, 0...], .. .. ]
#     """
#     edge_list = []
#     for f in face_data:
#         edge_list.append((f[0], f[1]) if f[0]<f[1] else (f[1], f[0]))
#         edge_list.append((f[1], f[2]) if f[1]<f[2] else (f[2], f[1]))
#         edge_list.append((f[2], f[0]) if f[2]<f[0] else (f[0], f[2]))
#     edge_list = list(set(edge_list))
#     # print(edge_list)
#     edge_mat = np.zeros((len(edge_list), num_vert))
#     for idx, e in enumerate(edge_list):
#         edge_mat[idx, e[0]]=1
#         edge_mat[idx, e[1]]=-1
#     return edge_mat

# U = []
# D = []
# import scipy.sparse
# for i in range(4):
#     u = scipy.sparse.load_npz(meta_dir+'upsampling_matrix'+str(i+1)+'.npz')
#     U.append(u)

# for i in range(4):
#     d = scipy.sparse.load_npz(meta_dir+'downsampling_matrix'+str(i+1)+'.npz')
#     D.append(d)

# print("sampling transforms", U, D)

# class SpiralConv(tf.keras.layers.Layer):
    
#     def __init__(self, in_channels, out_channels, indices, dim=1):
#         super(SpiralConv, self).__init__()
#         self.dim = dim
#         self.indices = indices
#         self.nodes = tf.shape(indices)[0]
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.inp_dense = self.in_channels*tf.shape(indices)[1]
#         self.layer = tf.keras.layers.Dense(out_channels)
        
#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#         x = tf.reshape(tf.gather(inputs, tf.reshape(self.indices, [-1]), axis=1), [batch_size, self.nodes, self.inp_dense])
#         return x
    
#     def model(self):
#         inputs = tf.keras.Input(shape=(self.nodes, self.in_channels))
#         x = self.call(inputs)
#         outputs = self.layer(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         return model

# in_channels = 2
# out_channels = 3
# batch_size = 1
# indices = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]
# nodes = tf.shape(indices)[0]
# inp_dense = in_channels*tf.shape(indices)[1]
# inp = [[[0, 0], [1, 1], [-1, -1]]]
# out = tf.reshape(tf.gather(inp, tf.reshape(indices, [-1]), axis=1), [batch_size, nodes, inp_dense])
# tf.print(out)

# class Upsampling(tf.keras.layers.Layer):
#     ## Sparse Mult code from coma
#     def __init__(self, upsampling_matrix):
#         super(Upsampling, self).__init__()
#         self.mat = upsampling_matrix
#         type(self.mat)
#         self.mat = self.mat.tocoo()
#         indices = np.column_stack((self.mat.row, self.mat.col))
#         self.mat = tf.sparse.SparseTensor(indices, self.mat.data, self.mat.shape)
#         self.mat = tf.sparse.reorder(self.mat)
#         self.Mp = self.mat.shape[0]
    
#     def call(self, inputs):
#         N = tf.shape(inputs)[0]
#         M = tf.shape(inputs)[1]
#         Fin = tf.shape(inputs)[2]
#         # N, M, Fin = int(N), int(M), int(Fin)

#         x = tf.transpose(inputs, perm=[1, 2, 0])  # M x Fin x N
#         x = tf.reshape(x, [M, -1])  # M x Fin*N
#         ##Speed up using sparse matrix multiplication
#         x = tf.sparse.sparse_dense_matmul(self.mat, x) # Mp x Fin*N
#         x = tf.reshape(x, [self.Mp, Fin, -1])  # Mp x Fin x N
#         x = tf.transpose(x, perm=[2,0,1]) # N x Mp x Fin
#         return x

#     # def compute_output_shape(self, input_shape):
#     #     new_shape = (input_shape[0], self.Mp,
#     #                  input_shape[2])
#     #     return new_shape

# upml1 = Upsampling(scipy.sparse.csc_matrix.astype(U[-1], dtype=np.float32))
# upml2 = Upsampling(scipy.sparse.csc_matrix.astype(U[-2], dtype=np.float32))

# # plot_vertices(mano.V_temp)
# # down_sampled = D[0] @ normalized_vertices[:,:3]
# # plot_vertices(down_sampled)
# # down_sampled = D[1] @ down_sampled
# # plot_vertices(down_sampled)

# # up_sampled = upml2.call(tf.convert_to_tensor([down_sampled], dtype=tf.float32))
# # plot_vertices(up_sampled[0])
# # up_sampled = upml1.call(tf.convert_to_tensor(up_sampled, dtype=tf.float32))
# # plot_vertices(up_sampled[0])

# faces = [[1, 2, 3], [3, 4, 1], [4, 3, 8], [8, 5, 4], [5, 8, 7], [7, 6, 5], [6, 7, 2], [2, 1, 6], [1, 4, 5], [5, 6, 1], [8, 3, 2], [2, 7, 8]]
# faces = list(np.array(faces)-1)
# edge_mat = get_edge_mat(faces, 8)
# print(edge_mat, edge_mat.shape)

# def get_sparse_edge_mat(edge_mat):
#     """
#     edge_mat: Num_edges_in_face*778 
#     """
#     edge_mat = scipy.sparse.csr_matrix(edge_mat)
#     edge_mat = edge_mat.tocoo()
#     indices = np.column_stack((edge_mat.row, edge_mat.col))
#     edge_mat = tf.SparseTensor(indices, edge_mat.data, edge_mat.shape)
#     edge_mat = tf.sparse.reorder(edge_mat)
#     return edge_mat

# tfspm = tf.convert_to_tensor(edge_mat, dtype=tf.float64)
# tfspm = get_sparse_edge_mat(edge_mat)
# tf.print(tfspm)
# ## Careful
# tensor_edge_mat = tfspm

# vertices = [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, -1], [0, 1, -1], [0, 0, -1], [1, 0, -1]]
# y_true = [vertices]
# y_true = tf.convert_to_tensor(y_true, dtype=tf.float64)

# y_pred = [np.array(vertices)-[0, 1, 0]]
# y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float64)

# def loss_function(y_true, y_pred):
#     num_verts = tf.shape(y_true)[1]
#     num_edges = tf.shape(tensor_edge_mat)[0]
#     y_true = tf.transpose(y_true, perm=[1, 2, 0]) # 778 x 3 x N
#     y_true = tf.reshape(y_true, [num_verts, -1])
#     y_pred = tf.transpose(y_pred, perm=[1, 2, 0]) # 778 x 3 x N
#     y_pred = tf.reshape(y_pred, [num_verts, -1])
#     edge_true = tf.sparse.sparse_dense_matmul(tensor_edge_mat, y_true) # num_edges x 3N
#     edge_pred = tf.sparse.sparse_dense_matmul(tensor_edge_mat, y_pred) # num_edges x 3N
#     edge_pred = tf.reshape(edge_pred, [num_edges, 3, -1])
#     edge_true = tf.reshape(edge_true, [num_edges, 3, -1])
#     y_pred = tf.reshape(y_pred, [num_verts, 3, -1]) # 778 x 3 x N
#     y_pred = tf.transpose(y_pred, perm=[2, 0, 1]) # N x 778 x 3
#     y_true = tf.reshape(y_true, [num_verts, 3, -1])
#     y_true = tf.transpose(y_true, perm=[2, 0, 1]) # N x 778 x 3

# #     tf.print(tf.shape(y_true), tf.shape(tf.reduce_sum(tf.norm(y_true - y_pred, ord=1, axis=1), axis=0)), tf.shape(tf.reduce_sum(tf.abs(tf.norm(edge_true, ord='euclidean', axis=1)**2 - tf.norm(edge_pred, ord='euclidean', axis=1)**2), axis=0)), output_stream=sys.stdout)
# #     tf.print(tf.norm(edge_true, ord='euclidean', axis=1) - tf.norm(edge_pred, ord='euclidean', axis=1) , output_stream=sys.stdout)
# #     tf.print(tf.reduce_mean(tf.norm(y_true - y_pred, ord=1, axis=1), axis=-1) , output_stream=sys.stdout)
# #     tf.print(tf.reduce_mean(tf.reduce_mean(tf.abs(tf.norm(edge_true, ord=2, axis=1) - tf.norm(edge_pred, ord=2, axis=1)), axis=-1), axis=-1))
# #     tf.print(tf.reduce_max(tf.abs(tf.norm(edge_true, ord=2, axis=1) - tf.norm(edge_pred, ord=2, axis=1))), output_stream=sys.stdout)
# #     tf.autograph.trace()
# #     tf.autograph.trace()

#     return tf.reduce_mean(tf.reduce_sum(tf.norm(y_true - y_pred, ord=1, axis=2), axis=1)) + tf.reduce_mean(tf.reduce_sum(tf.abs(tf.norm(edge_true, ord='euclidean', axis=1)**2 - tf.norm(edge_pred, ord='euclidean', axis=1)**2), axis=0))

# tf.print(loss_function(y_true, y_pred))

# sparse_edge_mat = tf.convert_to_tensor(get_edge_mat(mano.triangles, len(mano.V_temp)), dtype=tf.float32)

# tensor_edge_mat = get_sparse_edge_mat(sparse_edge_mat)

# vertices = mano.V_temp
# y_true = [vertices]
# y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

# y_pred = [np.array(vertices)-[0, 1, 0]]
# y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

# tf.print(loss_function(y_true, y_pred))

# class Error_Metric(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super(Error_Metric, self).__init__(**kwargs)
#         self.vertex_distance_error = self.add_weight(name="vertex_dist_error", initializer='zeros')
#         self.mean_error = self.add_weight(name="mean_vertex_dist_error", initializer='zeros')
#         self.steps = self.add_weight(name="steps", initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.vertex_distance_error = tf.reduce_mean(tf.reduce_sum(tf.norm(y_true - y_pred, ord='euclidean', axis=2), axis=1)) 
#         self.mean_error.assign_add(self.vertex_distance_error)
#         self.steps.assign_add(1)

#     def result(self):
#         return self.vertex_distance_error

#     def reset_states(self):
#         self.mean_error.assign(0.0)
#         self.steps.assign(0.0)

# with open(meta_dir+'indices'+str(1)+'.npy', 'rb') as f:
#     indices_1 = np.load(f)
# with open(meta_dir+'indices'+str(2)+'.npy', 'rb') as f:
#     indices_2 = np.load(f)
# with open(meta_dir+'indices'+str(3)+'.npy', 'rb') as f:
#     indices_3 = np.load(f)
# with open(meta_dir+'indices'+str(4)+'.npy', 'rb') as f:
#     indices_4 = np.load(f)

# def Print(x, name="def"):
#     tf.debugging.check_numerics(x, f'${name} FAILS')
#     # tf.print(name)
#     # tf.print(x)
#     return x

# resnet50 = tf.keras.applications.ResNet50(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000)

# up1 = Upsampling(scipy.sparse.csc_matrix.astype(U[0], dtype=np.float32))
# sconv1 = SpiralConv(48, 32, indices_1).model()
# up2 = Upsampling(scipy.sparse.csc_matrix.astype(U[1], dtype=np.float32))
# sconv2 = SpiralConv(32, 32, indices_2).model()
# up3 = Upsampling(scipy.sparse.csc_matrix.astype(U[2], dtype=np.float32))
# sconv3 = SpiralConv(32, 16, indices_3).model()
# up4 = Upsampling(scipy.sparse.csc_matrix.astype(U[3], dtype=np.float32))
# sconv4 = SpiralConv(16, 3, indices_4).model()

# I_input = tf.keras.Input(shape=(out_img_size, out_img_size, 3), name="I")
# proj_mat = tf.keras.Input(shape=(4, 4), name="proj_mat")
# affine_mat = tf.keras.Input(shape=(4, 4), name="affine_mat")

# x = resnet50(I_input)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# # x = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu , name="FC1")(x)
# # x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = tf.keras.layers.Dense(49*48, activation=tf.keras.activations.relu, name="FC2")(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = tf.keras.layers.Reshape((49, 48), name="reshape_to_mesh")(x)

# x = up1(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = sconv1(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = up2(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = sconv2(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = up3(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = sconv3(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = up4(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# x = sconv4(x)
# x = tf.keras.layers.Lambda(lambda y : Print(y))(x)

# V = tf.keras.layers.Reshape((n_verts_to_predict, 3), name="V")(x)

# annotations_xyz = tf.keras.layers.Lambda(lambda x: tf.py_function(func=unproject_freihand, inp=[*x], Tout=tf.float32), output_shape=(778, 3), name="annotations_xyz")([V, affine_mat, proj_mat])

# model = tf.keras.Model(inputs=[I_input, proj_mat, affine_mat], outputs=[V, annotations_xyz], name="basic_model")

# model.summary()


# ## Implement scheduler
# learning_rate = 1e-4
# boundaries = [80, 100, 160]
# lr_values = [learning_rate, learning_rate*0.1, learning_rate*0.01, learning_rate*0.001]

# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate),
#     loss={"V": loss_function},
#     metrics={"annotations_xyz": Error_Metric()}
# )

# model.fit(
#     ds_train, 
#     validation_data=ds_val,
#     epochs=500
# )



# # feature extractor
# I_input = tf.keras.Input(shape=(out_img_size, out_img_size, 3), name="I")
# proj_mat = tf.keras.Input(shape=(4, 4), name="proj_mat")
# affine_mat = tf.keras.Input(shape=(4, 4), name="affine_mat")

# # efficient_net_b0 = tf.keras.applications.EfficientNetB0(
# #     include_top=False, weights=None, input_shape=(out_img_size, out_img_size, 3)
# # )
# # efficient_net_features = efficient_net_b0(I_input)
# # flatten = tf.keras.layers.Flatten()(efficient_net_features)

# resnet50 = tf.keras.applications.ResNet50(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000)
# resnet_features = resnet50(I_input)

# flatten = tf.keras.layers.Flatten()(resnet_features)
# dense0 = tf.keras.layers.Dense(n_verts_to_predict*2)(flatten)
# relu0 = tf.keras.layers.ReLU()(dense0)
# dense1 = tf.keras.layers.Dense(n_verts_to_predict*3)(relu0)
# relu1 = tf.keras.layers.ReLU()(dense1)
# dense2 = tf.keras.layers.Dense(n_verts_to_predict*3)(relu1)
# V = tf.keras.layers.Reshape((n_verts_to_predict, 3), name="V")(dense2)

# annotations_xyz = tf.keras.layers.Lambda(lambda x: tf.py_function(func=unproject_freihand, inp=[*x], Tout=tf.float32), output_shape=(778, 3), name="annotations_xyz")([V, affine_mat, proj_mat])

# model = tf.keras.Model(inputs=[I_input, proj_mat, affine_mat], outputs=[V, annotations_xyz], name="basic_model")
# print(model.summary())

# learning_rate = 1e-3
# boundaries = [100, 130]
# lr_values = [learning_rate, learning_rate*0.1, learning_rate*0.01]

# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate),
#     loss={"V": tf.keras.losses.MeanAbsoluteError()},
#     metrics={"annotations_xyz": tf.keras.metrics.MeanAbsoluteError()}
# )

# # model.fit(
# #     ds_train1, 
# #     batch_size=1,
# #     validation_data=ds_val1,
# #     epochs=200
# # )
# model.fit(
#     ds_train, 
#     validation_data=ds_val,
#     epochs=300
# )
# #     callbacks=[
# #         tf.keras.callbacks.ReduceLROnPlateau(),
# #     ]

# example = next(iter(ds_train))
# V_pred_778x2 = model(example[0])[0]

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(I_normalize)

# ax2 = fig.add_subplot(122)
# ax2.scatter(V_pred_778x2[:, 0], V_pred_778x2[:, 1])
# # ax2.set_xlim(-1, 1)
# # ax2.set_ylim(1, -1)
# ax2.set_aspect("equal")

# # unnormalize
# V_pred_unnormalize_778x2 = V_pred_778x2.numpy().copy()

# shift_z = np.min(V_pred_unnormalize_778x2[:, 2])
# scale_z = np.max(V_pred_unnormalize_778x2[:, 2])-np.min(V_pred_unnormalize_778x2[:, 2])

# V_pred_unnormalize_778x2 = (V_pred_unnormalize_778x2 + 1) * (out_img_size/2)

# fig = plot_2d_pts_on_img(I_normalize, V_pred_unnormalize_778x2)
# fig.show()

# V_pred_778x2

# with open('./test1.json', 'w') as t:
#     json.dump({'pred':V_pred_778x2.numpy().tolist()}, t)

# y_true = [[1., 1], [0., 0.]]
# y_pred = [[1., 1.], [1., 0.]]
# # Using 'auto'/'sum_over_batch_size' reduction type.
# mae = tf.keras.losses.MeanAbsoluteError()
# mae(y_true, y_pred).numpy()


# model_ld = create_model()

# image = imageio.imread('../../../storage/sample_friehand/00000'+"000"[:(3-len(str(0)))]+str(0)+'.jpg')


