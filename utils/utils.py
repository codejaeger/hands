from types import SimpleNamespace
import chumpy as ch       # pip install git+https://github.com/scottandrews/chumpy.git@fe51783e0364bf1e9b705541e7d77f894dd2b1ac
import pickle
import numpy as np
import plotly.graph_objects as go   # for visualizing 3d
import plotly.express as px


def read_mano(path_to_pkl):
    with open(path_to_pkl, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    mano = SimpleNamespace()

    mano.V_temp = data["v_template"]
    mano.J_reg = data["J_regressor"].tocsr()    # initially it is in csc format
    mano.S = np.array(data["shapedirs"])
    mano.P = data["posedirs"]
    mano.W = data["weights"]

    mano.triangles = data["f"]
    mano.parent = data["kintree_table"][0].astype(np.int32)

    # pose pca paramaeters
    mano.pose_pca_basis = data["hands_components"]  # each row is a basis vector
    mano.pose_pca_mean = data["hands_mean"]
    mano.data_pose_pca_coeffs = data["hands_coeffs"]

    # helper constants
    mano.n_joints = mano.J_reg.shape[0]
    mano.n_vertices = mano.V_temp.shape[0]
    mano.n_triangles = mano.triangles.shape[0]

    return mano


def scatter3d(pts, size=1, color="red", opacity=0.5):
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        )
    )


def update_fig_range(fig, xrange, yrange, zrange):
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=xrange,),
            yaxis=dict(nticks=4, range=yrange,),
            zaxis=dict(nticks=4, range=zrange,)
        )
    )


def plot_2d_pts_on_img(img, pts):
    fig = px.imshow(img)
    fig.add_trace(
        go.Scatter(x=pts[:, 0], y=pts[:, 1],
        mode='markers',
        marker=dict(
                size=2,
                color="red",
                opacity=0.7,
                )
        )
    )
    
    return fig

