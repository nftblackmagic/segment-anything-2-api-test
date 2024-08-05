from flask import Flask, request, jsonify
from sam2.build_sam import build_sam2_video_predictor
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import urllib.request
import uuid
import json
import pickle
from flask import url_for

# At the top of your script, after importing Flask
app = Flask(__name__, static_folder='static')

# Create the static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Initialize the model
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def annotate_frame(frame_idx, frame_names, video_dir, points=None, labels=None, masks=None):
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title(f"frame {frame_idx}")
    
    # Load and display the image
    img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    ax.imshow(img)
    
    # Display points and masks if provided
    if points is not None and labels is not None:
        show_points(points, labels, ax)
    if masks is not None:
        for obj_id, mask in masks.items():
            show_mask(mask, ax, obj_id=obj_id)
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure to a buffer
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    
    # Save the figure locally for debugging
    debug_output_dir = "./debug_frames"
    os.makedirs(debug_output_dir, exist_ok=True)
    debug_frame_path = os.path.join(debug_output_dir, f"frame_{frame_idx}.png")
    plt.savefig(debug_frame_path)
    
    plt.close(fig)
    
    return img_data



@app.route('/upload_video', methods=['POST'])
def upload_video():
    data = request.json
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "Missing video_url parameter"}), 400

    # Generate a unique ID for this video processing session
    session_id = str(uuid.uuid4())
    video_dir = f"./temp_frames_{session_id}"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "input_video.mp4")

    try:
        # Download video
        urllib.request.urlretrieve(video_url, video_path)

        # Extract frames
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_dir, f"{count}.jpg"), image)
            success, image = vidcap.read()
            count += 1

        # Process video
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        # if torch.cuda.get_device_properties(0).major >= 8:
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        # inference_state = predictor.init_state(video_path=video_dir)
        
        # # Save inference_state
        # with open(f"./inference_state_{session_id}.pkl", 'wb') as f:
        #     pickle.dump(inference_state, f)

    except Exception as e:
        # Cleanup in case of an error
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                os.remove(os.path.join(video_dir, file))
            os.rmdir(video_dir)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "Video uploaded, frames extracted, and inference state initialized successfully",
        "session_id": session_id,
        "frame_count": count
    })
    
@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    
    # Check for required fields
    required_fields = ['session_id', 'points', 'labels', 'ann_frame_idx', 'ann_obj_id']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    session_id = data['session_id']
    points = np.array(data['points'], dtype=np.float32)
    labels = np.array(data['labels'], dtype=np.int32)
    ann_frame_idx = data['ann_frame_idx']
    ann_obj_id = data['ann_obj_id']

    video_dir = f"./temp_frames_{session_id}"
    
    if not os.path.exists(video_dir):
        return jsonify({"error": "Invalid session ID"}), 400

    try:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Load inference_state
        inference_state = predictor.init_state(video_path=video_dir)
    except FileNotFoundError:
        return jsonify({"error": "Inference state not found. Please upload the video first."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Check if ann_frame_idx is valid
    if ann_frame_idx < 0 or ann_frame_idx >= len(frame_names):
        return jsonify({"error": f"Invalid ann_frame_idx. Must be between 0 and {len(frame_names) - 1}"}), 400

    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Create output video
        output_video_path = f"static/segmented_video_{session_id}.mp4"
        first_frame = annotate_frame(0, frame_names, video_dir)
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            if out_frame_idx in video_segments:
                annotated_frame = annotate_frame(out_frame_idx, frame_names, video_dir, masks=video_segments[out_frame_idx])
            else:
                annotated_frame = annotate_frame(out_frame_idx, frame_names, video_dir)
            video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        video_writer.release()

        # Clean up
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)
        # os.remove(f"./inference_state_{session_id}.pkl")

        # Prepare output
        output_video_url = url_for('static', filename=f'segmented_video_{session_id}.mp4', _external=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "output_video_url": output_video_url
    })

if __name__ == '__main__':
    app.run(debug=True)
