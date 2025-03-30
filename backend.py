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
import scipy.io as sio
import plotly.io as pio
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import json
from functions import *

app = Flask(__name__)

# In backend.py, modify the process_image function

def process_image(file_path, net_path):
    V = load_image(file_path)

    net = torch.jit.load(net_path, map_location=torch.device('cpu'))
    n_2CH, p_2CH, n_3CH, p_3CH, n_4CH, p_4CH, n_SA, p_SA = cmr_planning(V, net)

    views_id = ['2-chamber', '3-chamber', '4-chamber', 'short-axis']
    
    views = [
        (p_2CH, n_2CH, '2-chamber'),
        (p_3CH, n_3CH, '3-chamber'),
        (p_4CH, n_4CH, '4-chamber'),
        (p_SA, n_SA, 'short-axis'),
    ]

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
        # Create a new plotly figure
        fig = go.Figure()
        
        # Add the grayscale image as a heatmap
        fig.add_trace(go.Heatmap(
            z=B[view].tolist(),  # Convert numpy array to list for JSON serialization
            colorscale='gray',
            showscale=False,
            x=np.linspace(extent[view][0], extent[view][1], B[view].shape[1]).tolist(),  # Convert to list
            y=np.linspace(extent[view][2], extent[view][3], B[view].shape[0]).tolist()   # Convert to list
        ))
        
        # Add each intersecting line as a separate trace
        for idx, (intersecting_line, view_id_other) in enumerate(intersections_2D[view]):
            fig.add_trace(go.Scatter(
                x=intersecting_line[:, 0].tolist(),  # Convert to list
                y=intersecting_line[:, 1].tolist(),  # Convert to list
                mode='lines',
                line=dict(color=plane_color[view_id_other], width=2),
                name=view_id_other
            ))
        
        # Layout configuration
        fig.update_layout(
            #title=view,
            autosize=False,  # Set to False to ensure dimensions are respected
            height=480,      # Slightly smaller than container
            width=480,       # Slightly smaller than container
            margin=dict(l=40, r=40, b=40, t=50, pad=4),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        )
        
        # Convert the figure to JSON - use plotly's to_json which handles numpy arrays correctly
        plotly_json = fig.to_json()
        
        slices.append({'view': view, 'plotly_json': plotly_json})

    return slices

@app.route('/')
def index():
    return render_template('index.html')

NET_PATH = "./models/best_model_complete.pth"

# flask route to handle DICOM uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dicom_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dicom_file']
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)

    # Process the DICOM file
    slices = process_image(file_path, NET_PATH)

    return jsonify({'slices': slices})

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/debug_json', methods=['GET'])
def debug_json():
    # Create a sample figure
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
    return jsonify({'sample_plotly': fig.to_json()})