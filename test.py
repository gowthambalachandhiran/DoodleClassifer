# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:41:04 2024

@author: gowtham.balachan
"""

import io
import unittest
from app import app

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        # Open a sample image file in binary mode
        with open('path_to_sample_image.png', 'rb') as img:
            img_data = img.read()

        # Send a POST request to the /predict endpoint with the image
        response = self.app.post('/predict', content_type='multipart/form-data', data={'image': (io.BytesIO(img_data), 'test.png')})
        
        # Check the status code
        self.assertEqual(response.status_code, 200)
        
        # Check the response data
        response_data = response.get_json()
        self.assertIn('predicted_class', response_data)
        self.assertIn('confidence', response_data)

if __name__ == '__main__':
    unittest.main()
