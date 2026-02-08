import pytest
import cv2
import os
import sys
import shutil
import numpy as np

# Ensure src folder is in path if candidorposed.py is in src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from candidorposed import classify_candid_or_posed, save_image, load_images

# -------------------------
# Test class for unit tests
# -------------------------
class TestCandidOrPosed:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup input and output folders
        self.test_input_dir = "input"
        self.test_output_dir = "output"
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Create a dummy image if input folder is empty
        if not os.listdir(self.test_input_dir):
            dummy_img = 255 * np.ones((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.test_input_dir, "dummy.jpg"), dummy_img)

        yield
        # Cleanup output folder after tests
        

    def test_classify_candid_or_posed_valid_image(self):
        filenames = [f for f in os.listdir(self.test_input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        assert len(filenames) > 0, "No images in input folder"

        for filename, img in load_images(self.test_input_dir, filenames):
            label, score, blur_var = classify_candid_or_posed(img)
            assert label in ["CANDID", "POSED"]
            assert isinstance(score, int)
            assert isinstance(blur_var, (float, np.floating))

            # Test saving
            output_path = save_image(self.test_output_dir, filename, img)
            assert os.path.exists(output_path)
            print(f"✓ Processed and saved: {output_path}")

    def test_classify_candid_or_posed_none_input(self):
        # Should raise an exception if None input
        with pytest.raises(cv2.error):
            classify_candid_or_posed(None)

    def test_load_images_skips_nonexistent(self):
        fake_files = ["nonexistent.jpg"]
        gen = load_images(self.test_input_dir, fake_files)
        assert list(gen) == [], "Should skip nonexistent images"

    def test_save_image_creates_file(self):
        img = 255 * np.ones((50, 50, 3), dtype=np.uint8)
        filename = "test_save.jpg"
        path = save_image(self.test_output_dir, filename, img, prefix="test_")
        assert os.path.exists(path)
        assert path.endswith("test_" + filename)
