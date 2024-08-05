import unittest
import requests
import json

class TestVideoProcessingAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5000"  # Adjust this if your server is hosted elsewhere
    
    def setUp(self):
        # This method will be called before each test
        self.upload_url = f"{self.BASE_URL}/upload_video"
        self.process_url = f"{self.BASE_URL}/process_video"
        self.session_id = None

    def test_video_processing_workflow(self):
        # Test uploading video url
        upload_data = {
            "video_url": "https://upcdn.io/FW25b7k/raw/uploads/test.mp4"
        }
        upload_response = requests.post(self.upload_url, json=upload_data)
        
        self.assertEqual(upload_response.status_code, 200, "Upload request failed")
        
        upload_result = upload_response.json()
        self.assertIn('session_id', upload_result, "Session ID not found in upload response")
        self.assertIn('frame_count', upload_result, "Frame count not found in upload response")
        
        self.session_id = upload_result['session_id']

        # Test processing video click points
        process_data = {
            "session_id": self.session_id,
            "points": [
                [
                    603,
                    866
                ]
            ],
            "labels": [
                1
            ],
            "ann_frame_idx": 0,
            "ann_obj_id": 1
        }
        process_response = requests.post(self.process_url, json=process_data)
        
        self.assertEqual(process_response.status_code, 200, "Process request failed")
        
        process_result = process_response.json()
        self.assertIn('output_video_url', process_result, "Output video URL not found in process response")
        # self.assertIn('output_data', process_result, "Output data not found in process response")
        
        # output_data = process_result['output_data']
        # self.assertIn('out_obj_ids', output_data, "Object IDs not found in output data")
        # self.assertIn('out_mask_logits', output_data, "Mask logits not found in output data")

    # def test_invalid_session_id(self):
    #     process_data = {
    #         "session_id": "invalid_session_id",
    #         "points": [[100, 100]],
    #         "labels": [1],
    #         "ann_frame_idx": 0,
    #         "ann_obj_id": 1
    #     }
    #     process_response = requests.post(self.process_url, json=process_data)
        
    #     self.assertEqual(process_response.status_code, 400, "Expected 400 status code for invalid session ID")

    # def test_missing_required_fields(self):
    #     # Test with missing 'points'
    #     process_data = {
    #         "session_id": "some_session_id",
    #         "labels": [1],
    #         "ann_frame_idx": 0,
    #         "ann_obj_id": 1
    #     }
    #     process_response = requests.post(self.process_url, json=process_data)
        
    #     self.assertEqual(process_response.status_code, 400, "Expected 400 status code for missing required field")
    #     self.assertIn("Missing required field", process_response.json().get('error', ''))

if __name__ == '__main__':
    unittest.main()