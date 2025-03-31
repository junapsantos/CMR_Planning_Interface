import os
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, jsonify
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import torch
import gc
import scipy.io as sio
import plotly.io as pio
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import json
from functions import *

app = Flask(__name__)

# Set a maximum file size (e.g., 100 MB) to prevent oversized uploads
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

NET_PATH = "./models/best_model_complete.pth"

def process_image(file_path, net_path):
    # Load the image using our load_image function
    V = load_image(file_path)

    # Load the model and set to eval mode
    net = torch.jit.load(net_path, map_location=torch.device('cpu'))
    net.eval()
    
    # Run the CNN planning function
    n_2CH, p_2CH, n_3CH, p_3CH, n_4CH, p_4CH, n_SA, p_SA = cmr_planning(V, net)

    # Prepare views for slicing
    views_id = ['2-chamber', '3-chamber', '4-chamber', 'short-axis']
    views = [
        (p_2CH, n_2CH, '2-chamber'),
        (p_3CH, n_3CH, '3-chamber'),
        (p_4CH, n_4CH, '4-chamber'),
        (p_SA, n_SA, 'short-axis'),
    ]

    # Compute oblique slices and intersections
    B, coords_3D, extent, intersections_2D, intersections_3D = oblique_slice(V, views)

    plane_color = {
        '2-chamber': 'green',
        '3-chamber': 'blue',
        '4-chamber': 'yellow',
        'short-axis': 'red'
    }

    slices = []
    # Create a Plotly figure for each view
    for view in views_id:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=B[view].tolist(),
            colorscale='gray',
            showscale=False,
            x=np.linspace(extent[view][0], extent[view][1], B[view].shape[1]).tolist(),
            y=np.linspace(extent[view][2], extent[view][3], B[view].shape[0]).tolist()
        ))
        for idx, (intersecting_line, view_id_other) in enumerate(intersections_2D[view]):
            fig.add_trace(go.Scatter(
                x=intersecting_line[:, 0].tolist(),
                y=intersecting_line[:, 1].tolist(),
                mode='lines',
                line=dict(color=plane_color[view_id_other], width=2),
                name=view_id_other
            ))
        fig.update_layout(
            autosize=False,
            height=480,
            width=480,
            margin=dict(l=40, r=40, b=40, t=50, pad=4),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        )
        plotly_json = fig.to_json()
        slices.append({'view': view, 'plotly_json': plotly_json})
    return slices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dicom_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dicom_file']
    filename = file.filename
    upload_dir = './uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    
    # Stream the file to disk in chunks to avoid memory overload
    with open(file_path, 'wb') as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    slices = process_image(file_path, NET_PATH)
    gc.collect()
    return jsonify({'slices': slices})

if __name__ == '__main__':
    app.run(debug=True)
