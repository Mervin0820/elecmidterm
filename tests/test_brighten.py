import pytest
import cv2
import os
import sys
import shutil

# Ensure src folder is in path if your brighten.py is in src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brighten import apply_clahe, save_image, load_images


class TestBrighten:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Use real input folder from repo
        self.test_input_dir = "input"
        self.test_output_dir = "output"
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        # Create a dummy image for testing if folder empty
        if not os.listdir(self.test_input_dir):
            dummy_img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.test_input_dir, "dummy.jpg"), dummy_img)
        yield
        # Optional cleanup: delete output folder after tests
        shutil.rmtree(self.test_output_dir)

    def test_apply_clahe_valid_image(self):
        filenames = [f for f in os.listdir(self.test_input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        assert len(filenames) > 0, "No images in input folder"
        for filename, img in load_images(self.test_input_dir, filenames):
            result = apply_clahe(img)
            assert result is not None
            # Output path check
            output_path = save_image(self.test_output_dir, filename, result)
            assert os.path.exists(output_path)
            print(f"✓ Processed and saved: {output_path}")

    def test_apply_clahe_none_input(self):
        with pytest.raises(ValueError, match="Input image is None"):
            apply_clahe(None)

    def test_load_images_skips_nonexistent(self):
        fake_files = ["nonexistent.jpg"]
        gen = load_images(self.test_input_dir, fake_files)
        assert list(gen) == [], "Should skip nonexistent images"

    def test_save_image_creates_file(self):
        import numpy as np
        img = 255 * np.ones((50, 50, 3), dtype=np.uint8)
        filename = "test_save.jpg"
        path = save_image(self.test_output_dir, filename, img, prefix="test_")
        assert os.path.exists(path)
        assert path.endswith("test_" + filename)
