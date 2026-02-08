import pytest
import cv2
import numpy as np
import os
import sys
import shutil

# Ensure src folder is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from beautify import smooth_skin, save_image, load_images

# -------------------------
# Test class for unit tests
# -------------------------
class TestBeautify:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Use real input folder from repo
        self.test_input_dir = "input"
        self.test_output_dir = "output"
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        yield
        # Keep input folder; do not delete output

    def test_smooth_skin_valid_image(self):
        filenames = [f for f in os.listdir(self.test_input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        assert len(filenames) > 0, "No images in input folder"
        for filename, img in load_images(self.test_input_dir, filenames):
            result = smooth_skin(img)
            assert result is not None
            output_path = save_image(self.test_output_dir, filename, result)
            assert os.path.exists(output_path)
            print(f"✓ Processed and saved: {output_path}")

    def test_smooth_skin_none_input(self):
        with pytest.raises(ValueError, match="Input image is None"):
            smooth_skin(None)
