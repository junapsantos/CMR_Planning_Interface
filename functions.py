import numpy as np
import scipy.ndimage
import torch
import pydicom
import scipy.io as sio
import torch.nn.functional as F

def resize_scan(scan, target_shape=(72, 72, 28)):
    """resize scan and aspect ratio for future redimensioning."""

    def reorder_axes(scan):
        """ reorder axes to ensure dimensions are being used to the fullest."""
        current_shape = np.array(scan.shape)

        sorted_axes = np.argsort(current_shape)
        scan_reordered = np.transpose(scan, axes=(sorted_axes[1], sorted_axes[2], sorted_axes[0]))

        new_axes = [sorted_axes[1], sorted_axes[2], sorted_axes[0]]

        return scan_reordered, new_axes
    
    scan, sorted_axes = reorder_axes(scan)

    scan_torch = torch.tensor(scan, dtype=torch.float32)

    current_shape = np.array(scan_torch.shape)
    target_shape = np.array(target_shape)

    scale_factors = target_shape / current_shape

    min_scale = min(1.0, scale_factors.min())
    new_shape = (current_shape * min_scale).astype(int)

    scan_tensor = scan_torch.unsqueeze(0).unsqueeze(0)  # add batch & channel dimensions
    scan_resized = F.interpolate(scan_tensor, size=tuple(new_shape), mode='trilinear', align_corners=False)
    scan_resized = scan_resized.squeeze(0).squeeze(0).numpy()

    # normalize after resizing
    scan_resized = (scan_resized - np.min(scan_resized)) / (np.max(scan_resized) - np.min(scan_resized))

    #print(f'Scan {scan.shape} --> {scan_resized.shape}')

    return scan_resized, scale_factors, sorted_axes

def preprocess_scan(scan, target_shape=(72, 72, 28)):
    """preprocess the scan by reorienting, resizing, and padding."""

    scan_resized, scale_factors, sorted_axes = resize_scan(scan, target_shape)

    resized_shape = np.array(scan_resized.shape)

    pad_values = [(0, max(target_shape[i] - resized_shape[i], 0)) for i in range(3)]
    scan_padded = np.pad(scan_resized, pad_values, mode='constant', constant_values=0)
    # crop to exactly match target size
    scan_cropped = scan_padded[:target_shape[0], :target_shape[1], :target_shape[2]]

    scan_tensor = torch.tensor(scan_cropped, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return scan_tensor, scale_factors, sorted_axes

def cmr_planning(scan, net):

    scan_processed, scale_factors, sorted_axes = preprocess_scan(scan)
    
    print('Scale Factors: ', scale_factors)
    net.eval()
    with torch.no_grad():
        cardiac_views = net(scan_processed)  

    def unreorder_vector(vector, inverse_axes):
        """reorder the vector components back to their original order."""

        vector_unordered = vector[inverse_axes]  

        return vector_unordered
    
    view_vectors = [[None, None, None] for _ in range(8)]

    for i in range(0, len(cardiac_views), 2):
        view_vectors[i + 1] = unreorder_vector(cardiac_views[i+1].squeeze().numpy() / scale_factors, inverse_axes=sorted_axes)
        normal_vector = unreorder_vector(cardiac_views[i].squeeze().numpy(), inverse_axes=sorted_axes)
        view_vectors[i] = normal_vector / np.linalg.norm(normal_vector)
        n_2CH, p_2CH, n_3CH, p_3CH, n_4CH, p_4CH, n_SA, p_SA = view_vectors

    return n_2CH, p_2CH, n_3CH, p_3CH, n_4CH, p_4CH, n_SA, p_SA


def clip_line_to_bounding_box(p0, line_dir, box_min, box_max):
    """clip line to stay within the bounding box."""
    t_vals = []
    for i in range(3): 
        min_t = (box_min[i] - p0[i]) / line_dir[i]
        max_t = (box_max[i] - p0[i]) / line_dir[i]
        if min_t > max_t:
            min_t, max_t = max_t, min_t
        t_vals.append((min_t, max_t))

    # intersection range by taking the maximum of min_t and the minimum of max_t
    t_min = max(t_vals[0][0], t_vals[1][0], t_vals[2][0])
    t_max = min(t_vals[0][1], t_vals[1][1], t_vals[2][1])

    return t_min, t_max



def oblique_slice(V, views, threshold = 1e-6):

    def plane_intersection(n1, d1, n2, d2):
        """ Computes the intersection line of two planes defined by normal vectors and distances. """
        # direction of intersection line (cross product of normal vectors)
        line_dir = np.cross(n1, n2)

        if np.linalg.norm(line_dir) < 1e-6:
            # planes are parallel
            p0 = None
            line_dir = None
        else:
            # point on the intersection line by solving:
            A = np.array([n1, n2, line_dir])
            d = np.array([d1, d2, 0])  # RHS of plane equations

            # solve for intersection point
            p0 = np.linalg.solve(A, d)

        return p0, line_dir
    
    B = {}
    coords_3D = {}
    extent = {}

    intersections_2D = {}
    intersections_3D = {}

    for point, normal, view_id in views:

        intersections_2D[view_id] = []
        intersections_3D[view_id] = []

        point = np.asarray(point).flatten()
        normal = np.asarray(normal).flatten()
        normal /= np.linalg.norm(normal)

        ref_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(normal, ref_vector)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)

        def project_to_slice(p):
            return np.array([np.dot(p - point, v1), np.dot(p - point, v2)])

        dims = np.array(V.shape)
        y_max, x_max, z_max = dims[0]-1, dims[1]-1, dims[2]-1

        # volume corners in world coordinates (x,y,z)
        volume_corners = np.array([
            [0,    0,    0],
            [x_max,0,    0],
            [0,    y_max,0],
            [0,    0,    z_max],
            [x_max,y_max,0],
            [x_max,0,    z_max],
            [0,    y_max,z_max],
            [x_max,y_max,z_max]
        ])

        corners_uv = np.array([project_to_slice(c) for c in volume_corners])
        max_extent = np.max(np.abs(corners_uv), axis=0)

        grid_size = max_extent * 2

        u_vals = np.arange(-grid_size[0] / 2, grid_size[0] / 2 + 1.0, 1.0)
        v_vals = np.arange(-grid_size[1] / 2, grid_size[1] / 2 + 1.0, 1.0)
        slice_u, slice_v = np.meshgrid(u_vals, v_vals)

        extent[view_id] = [slice_u.min(), slice_u.max(), slice_v.min(), slice_v.max()]

        # 3D coordinates for every (u,v) in the slice:
        coords_3D_aux = point[:, None, None] + slice_u * v1[:, None, None] + slice_v * v2[:, None, None]
        
        coords_3D[view_id] = coords_3D_aux

        x = coords_3D_aux[0]
        y = coords_3D_aux[1]
        z = coords_3D_aux[2]

        valid_mask = (x >= 0) & (x <= x_max) & (y >= 0) & (y <= y_max) & (z >= 0) & (z <= z_max)

        x_clamped = np.clip(x, 0, x_max)
        y_clamped = np.clip(y, 0, y_max)
        z_clamped = np.clip(z, 0, z_max)

        coords_flat = np.vstack([y_clamped.ravel(), x_clamped.ravel(), z_clamped.ravel()])

        slice_img = scipy.ndimage.map_coordinates(V, coords_flat, order=1,
                                                mode='constant', cval=0.0)
        slice_img = slice_img.reshape(slice_u.shape)
        slice_img[~valid_mask] = 0.0

        B_aux = slice_img

        B[view_id] = B_aux
        
        for point_other, normal_other, view_id_other in views:

            n_oblique, d_oblique = normal, np.dot(normal, point)
            n_other = np.asarray(normal_other).flatten()
            d_other = np.dot(n_other, point_other)

            p0, line_dir = plane_intersection(n_oblique, d_oblique, n_other, d_other)

            if p0 is not None:
                t_min, t_max = clip_line_to_bounding_box(p0, line_dir, [0, 0, 0], [x_max, y_max, z_max])

                # generate the intersection points within the valid range of t
                t_vals = np.linspace(t_min, t_max, 200)
                line_points = np.array([p0 + t * line_dir for t in t_vals])
                valid_line = np.all((line_points >= 0) & (line_points <= np.array([x_max, y_max, z_max])), axis=1)

                intersections_3D[view_id].append((line_points[valid_line], view_id_other))

                intersecting_line_2D = np.array([project_to_slice(p) for p in line_points[valid_line]])
                
                intersections_2D[view_id].append((intersecting_line_2D, view_id_other))
    
    return B, coords_3D, extent, intersections_2D, intersections_3D


def extract_image(data):
    """search for a 3D image inside the given data."""
    if isinstance(data, np.ndarray) and data.ndim in [2, 3]:
        return data if data.ndim == 3 else np.expand_dims(data, axis=-1)
    elif isinstance(data, dict):
        for key, value in data.items():
            img = extract_image(value)
            if img is not None:
                return img
    elif isinstance(data, list):
        for item in data:
            img = extract_image(item)
            if img is not None:
                return img
    return None

def load_image(file_path):
    if file_path.endswith(".dcm"):  # DICOM
        ds = pydicom.dcmread(file_path)
        img = None
        for attr in ['pixel_array', 'Pixels', 'PixelData']:
            if hasattr(ds, attr):
                img = getattr(ds, attr)
                break
        if img is None:
            raise ValueError("No valid image found in the DICOM file.")
    elif file_path.endswith(".npy"): 
        data = np.load(file_path, allow_pickle=True)
        img = extract_image(data)
    elif file_path.endswith(".mat"):  
        mat_contents = sio.loadmat(file_path)
        img = extract_image(mat_contents)
    
    # ensure image is at least 3D
    if img is not None and img.ndim == 2:
        img = np.expand_dims(img, axis=-1)  # Convert 2D to 3D by adding depth dimension
    elif img is None or img.ndim > 3:
        raise ValueError("No valid 3D image found in the input data.")
    
    return img