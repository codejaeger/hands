# Global Constants
EPSILON = 0.0001
AUGMENT = 1
PAD = 30
SEQ_LENGTH = 8              ##Spiral convolulational neighbourhood length
OUT_IMAGE_SIZE = (224, 224) ## Paper says 192x192 crops

CROP_OFFSET_RANGE = [-5, 5]
ROTATION_RANGE = [-180, 180]

## Hyperparameters
lamb_edge = 0.01
lamb_vert = 0.01
learning_rate = 1e-3
boundaries = [100, 150, 200, 250]
lr_values = [learning_rate, learning_rate*0.1, learning_rate*0.01, learning_rate*0.001, learning_rate*0.0001]

storage_dir = "./drive/MyDrive/data/"
log_dir = storage_dir + "logs/"

imagenet_mean=np.array([0.485, 0.456, 0.406])
imagenet_std=np.array([0.229, 0.224, 0.225])


def bounding_box(points, augment=1):
    """
    Get the hand's bounding box based on annotation coordinates
    """
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    random_offsets = np.random.randint(*CROP_OFFSET_RANGE, (2, 3))
    return np.array([[min(x_coordinates), min(y_coordinates), min(z_coordinates)], [max(x_coordinates), max(y_coordinates), max(z_coordinates)]])+random_offsets*augment

def translation_mat(tx=0, ty=0, tz=0):
    """
    4x4 translational matrix
    """
    mat = np.eye(4)
    mat[0:3, 3] = np.array([tx, ty, tz]).T
    return mat

def scaling_mat(sx=1, sy=1, sz=1):
    """
    4x4 scaling matrix
    """
    return np.diag([sx, sy, sz, 1])

def shift_vertices(vertex_set, augment=1, pad=PAD):
    """
    Shift vertices based on the crop of hand from image
    """
    bbx = bounding_box(vertex_set).astype(int)
    mat = translation_mat(*(-1*(bbx[0,:]-pad)))
#     print(mat)
    adjusted_vertices = np.hstack([vertex_set.copy(), np.ones((len(vertex_set), 1))])
    adjusted_vertices = (adjusted_vertices @ mat.T)
    return adjusted_vertices, mat, bbx

def flip_hand(adjusted_vertices, mat, cropped_image, is_left):
    """
    Flip left hand to mirrored right hand coordinates
    """
    flip_mat = np.eye(4)
    if is_left:
        cropped_image = cropped_image[:, -1::-1, :]
        flip_mat = translation_mat(cropped_image.shape[1])
        flip_mat[0, 0] = -1
        adjusted_vertices = adjusted_vertices @ flip_mat.T
    return cropped_image, adjusted_vertices, flip_mat

def crop_image(image, width, height, bbx, pad=PAD):
    image = np.asarray(image.resize((width, height), 2))
    cropped_image = image[bbx[0][1]-pad:bbx[1][1]+pad, bbx[0][0]-pad:bbx[1][0]+pad, :]
    return cropped_image

def visualize_in_orig_image(image_path, width, height, annotations):
    """
    Visualize hand annotations and image on uncropped image
    """
    image = Image.open(image_path)
    image = image.resize((width, height), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    vertices = np.array(annotations['vertices'])
    plt.plot(vertices[:, 0], vertices[:, 1], 'o', color='green', markersize=1)
    plt.show()

def visualize_in_processed_image(image_path, width, height, annotations):
    """
    Visualize hand annotations and image on cropped image
    """
    image = Image.open(image_path)
    adjusted_vertices, mat, bbx = shift_vertices(annotations['vertices'])
    cropped_image = crop_image(image, width, height, bbx)
    plt.imshow(cropped_image)
    vertices = np.array(adjusted_vertices)
    plt.plot(adjusted_vertices[:, 0], adjusted_vertices[:, 1], 'o', color='green', markersize=1)
    plt.show()

# def viz_3d(vertices, faces):
#     import open3d as o3d
    
#     mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
#     o3d.visualization.draw_geometries([mesh])

def plot_and_show(image, vertex_set=None):
    """
    Visualize already cropped image and annotations
    """
    plt.imshow(image)
    if vertex_set is not None:
        plt.plot(vertex_set[:, 0], vertex_set[:, 1], 'o', color='green', markersize=1)
    plt.show()