import requests
import json
import sys
import base64
from PIL import Image
import io
import os

BASE_URL = "http://127.0.0.1:5000"  # Adjust this if your server is hosted elsewhere
UPLOAD_URL = f"{BASE_URL}/upload_video"
PROCESS_URL = f"{BASE_URL}/process_image"

def process_image(input_file):
    # Read input from JSON file
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # Extract data from input
    video_url = input_data['video_source']
    clicks = input_data['clicks']

    # Upload video
    upload_data = {
        "video_url": video_url
    }
    upload_response = requests.post(UPLOAD_URL, json=upload_data)

    if upload_response.status_code != 200:
        print(f"Upload request failed with status code {upload_response.status_code}")
        return

    upload_result = upload_response.json()
    session_id = upload_result.get('session_id')
    frame_count = upload_result.get('frame_count')

    if not session_id:
        print("Session ID not found in upload response")
        return

    print(f"Video uploaded successfully. Session ID: {session_id}, Frame count: {frame_count}")

    # Create a directory to save output images
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Process image for each click
    for i, click in enumerate(clicks):
        process_data = {
            "session_id": session_id,
            "frame_idx": click['ann_frame_idx'],  # Use ann_frame_idx as frame_idx
            "points": click['points'],
            "labels": click['labels'],
            "ann_obj_id": click['ann_obj_id']
        }
        process_response = requests.post(PROCESS_URL, json=process_data)

        if process_response.status_code != 200:
            print(f"Process request failed with status code {process_response.status_code}")
            continue

        process_result = process_response.json()
        output_image_base64 = process_result.get('output_image')

        if output_image_base64:
            # Decode base64 string to image
            img_data = base64.b64decode(output_image_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # Save the image
            output_filename = f"output_image_{session_id}_click_{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path)
            print(f"Image processed successfully. Saved as: {output_path}")
        else:
            print("Output image data not found in process response")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_image(input_file)
