##########################################
## Patch file to generate a dataset of 3D hand mesh annoations of dimensions 778x3
## Referenced from - Github Link
##########################################
from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import argparse

from utils.fh_utils import *

def show_training_samples(base_path, version, num2show=None, render_mano=False):
    if render_mano:
        from utils.model import HandModel, recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all

    # load annotations
    db_data_anno = list(load_db_annotation(base_path, 'training'))

    anns = []

    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        msk = read_msk(idx, base_path)

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)
        print((K@np.hstack([xyz[:, :2], np.ones((21, 1))]).T).T[:1, :], uv[:1, :])
        K = np.hstack([K, np.zeros((3, 1))])
        K = np.vstack([K, [0, 0, 0, 1]])
        
        s_x = np.std(uv[:, 0]) / np.std(xyz[:, 0])
        K[2, 2] = s_x
        z_cam = s_x *(xyz[:, 2]).reshape(-1, 1)
        V = np.concatenate((uv[:, :2], z_cam), axis=1)
        print(V.shape)
        unp_z = V[:, 2] / s_x
        K_unp = [[1/K[0, 0], 0, -K[0, 2]/K[0, 0]], [0, 1/K[1, 1], -K[1, 2]/K[1,1 ]], [0, 0, 1]]
        V_unp = (K_unp @ np.hstack([V[:, :2], np.ones((21, 1))]).T).T
        V_unp[:, 2] = unp_z
        V_unp[:, 0] = V_unp[:, 0] * unp_z
        V_unp[:, 1] = V_unp[:, 1] * unp_z
        print("Imp", V_unp[:1, :], xyz[:1, :]) 
        V = np.hstack([V, np.ones((21, 1))])
        Vo = np.linalg.inv(K) @ V.T
        
        print(Vo.T[:3, :3], xyz[:3, :])
        # print(V.shape)
        # render an image of the shape
        msk_rendered = None
        print(idx)
        if render_mano:
            # split mano parameters
            # print(mano)
            poses, shapes, uv_root, scale = split_theta(mano)
            focal, pp = get_focal_pp(K)
            xyz_root = recover_root(uv_root, scale, focal, pp)

            # set up the hand model and feed hand parameters
            renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
            renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
            msk_rendered, V1 = renderer.render(K, img_shape=img.shape[:2])

        # show
        # fig = plt.figure()
        proj = projectPoints(V1, K)
        print(proj.shape)        
        s_x = np.std(proj[:, 0]) / np.std(V1[:, 0])
        z_cam = s_x *(V1[:, 2]).reshape(-1, 1)
        V2 = np.concatenate((proj[:, :2], z_cam), axis=1)

        
        
        
        anns.append(V2.tolist())
        
        # ax1 = fig.add_subplot(121)
        # ax1.plot(V2[:, 0], V2[:, 1], 'o', color='green', markersize=1)
        # # ax2 = fig.add_subplot(122)
        # ax1.imshow(img)
        # # ax2.imshow(msk if msk_rendered is None else msk_rendered)
        # # plot_hand(ax1, uv, order='uv')
        # # plot_hand(ax2, uv, order='uv')
        # # ax1.axis('off')
        # # ax2.axis('off')
        # # print(fig)
        # # plt.show()
        # fig.canvas.draw()
        # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # # plt.savefig('vis3d.png')
        # print(V1.shape)
        # from PIL import Image
        # im = Image.fromarray(image_from_plot)
        # im.save("rescale.png")
    dc = {}
    dc['anns'] = anns
    with open('anno.json', 'w') as f:
        json.dump(dc, f)
    # with open('anno.json', 'r') as f:
    #     l = json.load(f)
    # print(np.array(l['anns'][0]).shape)


def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in  range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        print(img)
        ax1.axis('off')
        plt.show()


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    # parser.add_argument('base_path', type=str,
    #                     help='Path to where the FreiHAND dataset is located.')
    # parser.add_argument('--show_eval', action='store_true',
    #                     help='Shows samples from the evaluation split if flag is set, shows training split otherwise.')
    # parser.add_argument('--mano', action='store_true',
    #                     help='Enables rendering of the hand if mano is available. See README for details.')
    # parser.add_argument('--num2show', type=int, default=-1,
    #                     help='Number of samples to show. ''-1'' defaults to show all.')
    # parser.add_argument('--sample_version', type=str, default=sample_version.gs,
    #                     help='Which sample version to use when showing the training set.'
    #                          ' Valid choices are %s' % sample_version.valid_options())
    # args = parser.parse_args()

    # check inputs
    # msg = 'Invalid choice: ''%s''. Must be in %s' 
    #% (args.sample_version, sample_version.valid_options())
    # assert args.sample_version in sample_version.valid_options(), msg

    # if args.show_eval:
    #     """ Show some evaluation samples. """
    #     show_eval_samples(args.base_path,
    #                       num2show=args.num2show)

    # else:
        """ Show some training samples. """
show_training_samples(
    '/home/hands/data/storage/',
    sample_version.gs,
    num2show=1,
    render_mano=True
)