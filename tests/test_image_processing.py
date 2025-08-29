import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pandas as pd
from carde.image_processing import pixel_size_um
from carde.image_processing import get_metadata_index
from carde.image_processing import get_metadata_index, crop_away_metadata
from carde.image_processing import preprocess_image
from carde.image_processing import preprocess_images, preprocess_image
from carde.image_processing import segment_otsu
from carde.image_processing import combine_images
from carde.image_processing import segment_combined
from carde.image_processing import combine_segmentation_with_overlay
from carde.image_processing import evaluate_segmentation
from carde.image_processing import process_folder


# Test cases for get_metadata_index


def test_get_metadata_index_no_metadata():
    # Create an image with no metadata (all zeros)
    image = np.zeros((100, 100), dtype=np.uint8)
    assert get_metadata_index(image) == -1


def test_get_metadata_index_with_metadata():
    # Create an image with metadata (a bright object at the bottom)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[90:, :] = 255
    assert get_metadata_index(image) == 89


def test_get_metadata_index_with_multiple_objects():
    # Create an image with multiple bright objects
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:19, :] = 255
    image[50:60, :] = 255
    image[85:, :] = 255
    assert get_metadata_index(image) == 84


def test_get_metadata_index_with_noise():
    # Create an image with noise and a bright object at the bottom
    np.random.seed(0)
    image = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image[90:, :] = 255
    assert get_metadata_index(image) == 89


def test_get_metadata_index_small_image():
    # Create a small image with a bright object
    image = np.zeros((20, 20), dtype=np.uint8)
    image[10:, :] = 255
    assert get_metadata_index(image) == 9


def test_get_metadata_index_less_than_10_lines():
    # Create a small image with a bright object
    image = np.zeros((20, 20), dtype=np.uint8)
    image[12:, :] = 255
    assert get_metadata_index(image) == -1


# Test cases for crop_away_metadata


def test_crop_away_metadata_no_metadata():
    # Create an image with no metadata (all zeros)
    image = np.zeros((100, 100), dtype=np.uint8)
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (100, 100)


def test_crop_away_metadata_with_metadata():
    # Create an image with metadata (a bright object at the bottom)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[90:, :] = 255
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (89, 100)


def test_crop_away_metadata_with_multiple_objects():
    # Create an image with multiple bright objects
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:19, :] = 255
    image[50:60, :] = 255
    image[85:, :] = 255
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (84, 100)


def test_crop_away_metadata_with_noise():
    # Create an image with noise and a bright object at the bottom
    np.random.seed(0)
    image = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image[90:, :] = 255
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (89, 100)


def test_crop_away_metadata_small_image():
    # Create a small image with a bright object
    image = np.zeros((20, 20), dtype=np.uint8)
    image[10:, :] = 255
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (9, 20)


def test_crop_away_metadata_less_than_10_lines():
    # Create a small image with a bright object
    image = np.zeros((20, 20), dtype=np.uint8)
    image[12:, :] = 255
    cropped_image = crop_away_metadata(image)
    assert cropped_image.shape == (20, 20)


# Test cases for preprocess_image


def setup_mocks(
    mock_imread, mock_get_metadata_index, mock_crop_away_metadata, image_shape, metadata_index, label_image_shape=None
):
    mock_image = np.zeros(image_shape, dtype=np.uint8)
    mock_imread.side_effect = (
        [mock_image] if label_image_shape is None else [mock_image, np.zeros(label_image_shape, dtype=np.uint8)]
    )
    mock_get_metadata_index.return_value = metadata_index
    mock_crop_away_metadata.side_effect = (
        [mock_image[:metadata_index, :]]
        if label_image_shape is None
        else [mock_image[:metadata_index, :], np.zeros(label_image_shape, dtype=np.uint8)[:metadata_index, :]]
    )
    return mock_image


@patch("carde.image_processing.imread")
@patch("carde.image_processing.get_metadata_index")
@patch("carde.image_processing.crop_away_metadata")
def test_preprocess_image_no_label(mock_crop_away_metadata, mock_get_metadata_index, mock_imread):
    mock_image = setup_mocks(mock_imread, mock_get_metadata_index, mock_crop_away_metadata, (100, 100), 89)

    mock_path = MagicMock(spec=Path)
    mock_path.parent.parent.glob.return_value.__next__.return_value.exists.return_value = False

    cropped_image, cropped_label_image = preprocess_image(mock_path)

    mock_imread.assert_called_once_with(mock_path)
    mock_get_metadata_index.assert_called_once_with(mock_image)
    mock_crop_away_metadata.assert_called_once_with(mock_image, 89)
    assert cropped_image.shape == (89, 100)
    assert cropped_label_image is None


@patch("carde.image_processing.imread")
@patch("carde.image_processing.get_metadata_index")
@patch("carde.image_processing.crop_away_metadata")
def test_preprocess_image_with_label(mock_crop_away_metadata, mock_get_metadata_index, mock_imread):
    mock_image = setup_mocks(
        mock_imread, mock_get_metadata_index, mock_crop_away_metadata, (100, 100), 89, (100, 100, 3)
    )

    mock_path = MagicMock(spec=Path)
    mock_path.parent.parent.glob.return_value.__next__.return_value.exists.return_value = True

    cropped_image, cropped_label_image = preprocess_image(mock_path)

    assert mock_imread.call_count == 2
    mock_get_metadata_index.assert_called_once_with(mock_image)
    assert mock_crop_away_metadata.call_count == 2
    assert cropped_image.shape == (89, 100)
    assert cropped_label_image.shape == (89, 100)


@patch("carde.image_processing.imread")
@patch("carde.image_processing.get_metadata_index")
@patch("carde.image_processing.crop_away_metadata")
def test_preprocess_image_label_shape_mismatch(mock_crop_away_metadata, mock_get_metadata_index, mock_imread):
    mock_image = setup_mocks(
        mock_imread, mock_get_metadata_index, mock_crop_away_metadata, (100, 100), 89, (50, 100, 3)
    )

    mock_path = MagicMock(spec=Path)
    mock_path.parent.parent.glob.return_value.__next__.return_value.exists.return_value = True

    with pytest.raises(AssertionError):
        preprocess_image(mock_path)


# Test cases for preprocess_images


@patch("carde.image_processing.imsave")
@patch("carde.image_processing.preprocess_image")
@patch("carde.image_processing.Path.glob")
@patch("carde.image_processing.Path.mkdir")
def test_preprocess_images_no_labels(mock_mkdir, mock_glob, mock_preprocess_image, mock_imsave):
    # Setup mock paths
    mock_image_path = MagicMock(spec=Path)
    mock_image_path.name = "image1.tif"
    mock_image_path.stem = "image1"
    mock_image_path.parent.parent.glob.return_value = [mock_image_path]
    mock_glob.return_value = [mock_image_path]

    # Setup mock preprocess_image return value
    mock_preprocess_image.return_value = (np.zeros((100, 100), dtype=np.uint8), None)

    # Call the function
    preprocess_images(
        data_path=Path("/mock/data"),
        target_image_path=Path("/mock/target/images"),
        target_label_path=Path("/mock/target/labels"),
    )

    # Assertions
    mock_glob.assert_called_once_with("[0-9]*/Original*/*.tif")
    mock_preprocess_image.assert_called_once_with(mock_image_path)
    mock_mkdir.assert_any_call(exist_ok=True, parents=True)
    args, kwargs = mock_imsave.call_args_list[0]
    assert args[0] == Path("/mock/target/images/image1.tif")
    np.testing.assert_array_equal(args[1], np.zeros((100, 100), dtype=np.uint8))
    assert mock_imsave.call_count == 1


@patch("carde.image_processing.imsave")
@patch("carde.image_processing.preprocess_image")
@patch("carde.image_processing.Path.glob")
@patch("carde.image_processing.Path.mkdir")
def test_preprocess_images_with_labels(mock_mkdir, mock_glob, mock_preprocess_image, mock_imsave):
    # Setup mock paths
    mock_image_path = MagicMock(spec=Path)
    mock_image_path.name = "image1.tif"
    mock_image_path.stem = "image1"
    mock_image_path.parent.parent.glob.return_value = [mock_image_path]
    mock_glob.return_value = [mock_image_path]

    # Setup mock preprocess_image return value
    mock_preprocess_image.return_value = (np.zeros((100, 100), dtype=np.uint8), np.ones((100, 100), dtype=np.uint8))

    # Call the function
    preprocess_images(
        data_path=Path("/mock/data"),
        target_image_path=Path("/mock/target/images"),
        target_label_path=Path("/mock/target/labels"),
    )

    # Assertions
    mock_glob.assert_called_once_with("[0-9]*/Original*/*.tif")
    mock_preprocess_image.assert_called_once_with(mock_image_path)
    mock_mkdir.assert_any_call(exist_ok=True, parents=True)
    args, kwargs = mock_imsave.call_args_list[0]
    assert args[0] == Path("/mock/target/images/image1.tif")
    np.testing.assert_array_equal(args[1], np.zeros((100, 100), dtype=np.uint8))
    args, kwargs = mock_imsave.call_args_list[1]
    assert args[0] == Path("/mock/target/labels/image1_label.png")
    np.testing.assert_array_equal(args[1], np.uint8(np.ones((100, 100), dtype=np.uint8) * 255))
    assert mock_imsave.call_count == 2


@patch("carde.image_processing.imsave")
@patch("carde.image_processing.preprocess_image")
@patch("carde.image_processing.Path.glob")
@patch("carde.image_processing.Path.mkdir")
def test_preprocess_images_multiple_images(mock_mkdir, mock_glob, mock_preprocess_image, mock_imsave):
    # Setup mock paths
    mock_image_path1 = MagicMock(spec=Path)
    mock_image_path1.name = "image1.tif"
    mock_image_path1.stem = "image1"
    mock_image_path2 = MagicMock(spec=Path)
    mock_image_path2.name = "image2.tif"
    mock_image_path2.stem = "image2"
    mock_glob.return_value = [mock_image_path1, mock_image_path2]

    # Setup mock preprocess_image return value
    mock_preprocess_image.side_effect = [
        (np.zeros((100, 100), dtype=np.uint8), None),
        (np.zeros((100, 100), dtype=np.uint8), np.ones((100, 100), dtype=np.uint8)),
    ]

    # Call the function
    preprocess_images(
        data_path=Path("/mock/data"),
        target_image_path=Path("/mock/target/images"),
        target_label_path=Path("/mock/target/labels"),
    )

    # Assertions
    mock_glob.assert_called_once_with("[0-9]*/Original*/*.tif")
    assert mock_preprocess_image.call_count == 2
    mock_mkdir.assert_any_call(exist_ok=True, parents=True)
    assert mock_imsave.call_count == 3


# Test cases for segment_otsu


def test_segment_otsu_basic():
    # Create a simple binary image with a single object
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255
    segmented = segment_otsu(image)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) == 20 * 20


def test_segment_otsu_with_noise():
    # Create an image with noise and a single object
    np.random.seed(0)
    image = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255
    segmented = segment_otsu(image)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) == 20 * 20


def test_segment_otsu_with_multiple_objects():
    # Create an image with multiple objects
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:30, 10:30] = 255
    image[70:90, 70:90] = 255
    segmented = segment_otsu(image)
    assert segmented.max() == 2
    assert segmented.min() == 0
    assert np.sum(segmented == 1) == 20 * 20
    assert np.sum(segmented == 2) == 20 * 20


def test_segment_otsu_with_small_objects():
    # Create an image with small objects
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:11, 10:11] = 255
    image[70:71, 70:71] = 255
    # minimum_size=6 because the object gets blurred by the Gaussian filter
    segmented = segment_otsu(image, minimum_size=6)
    assert segmented.max() == 0
    assert segmented.min() == 0
    assert np.sum(segmented) == 0


def test_segment_otsu_with_large_radius():
    # Create an image with a single object and use a large radius for background removal
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255
    segmented = segment_otsu(image, radius=50)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) > 0


def test_segment_otsu_with_different_sigma():
    # Create an image with a single object and use a different sigma for Gaussian filter
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255
    segmented = segment_otsu(image, sigma=2.0)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) > 0


def test_segment_otsu_with_no_objects():
    # Create an image with no objects
    image = np.zeros((100, 100), dtype=np.uint8)
    segmented = segment_otsu(image)
    assert segmented.max() == 0
    assert segmented.min() == 0
    assert np.sum(segmented) == 0


# Test cases for segment_combined


def test_segment_combined_basic():
    # Create simple binary images with a single object
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[41:61, 41:61] = 255
    segmented = segment_combined(image_SE2, image_inlens)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert (segmented[41:60, 41:60] == 1).all()


def test_segment_combined_with_noise():
    # Create images with noise and a single object
    np.random.seed(0)
    image_SE2 = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image_inlens = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[41:61, 41:61] = 255
    segmented = segment_combined(image_SE2, image_inlens)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) == 429


def test_segment_combined_with_multiple_objects():
    # Create images with multiple objects
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[10:30, 10:30] = 255
    image_inlens[70:90, 70:90] = 255
    segmented = segment_combined(image_SE2, image_inlens)
    assert segmented.max() == 2
    assert segmented.min() == 0
    assert np.sum(segmented == 1) == 20 * 20
    assert np.sum(segmented == 2) == 20 * 20


def test_segment_combined_with_small_objects():
    # Create images with small objects
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[10:11, 10:11] = 255
    image_inlens[70:71, 70:71] = 255
    # minimum_size=6 because the object gets blurred by the Gaussian filter
    segmented = segment_combined(image_SE2, image_inlens, minimum_size=6)
    assert segmented.max() == 0
    assert segmented.min() == 0
    assert np.sum(segmented) == 0


def test_segment_combined_with_large_radius():
    # Create images with a single object and use a large radius for background removal
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[40:60, 40:60] = 255
    segmented = segment_combined(image_SE2, image_inlens, radius=50)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) > 0


def test_segment_combined_with_different_sigma():
    # Create images with a single object and use a different sigma for Gaussian filter
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[40:60, 40:60] = 255
    segmented = segment_combined(image_SE2, image_inlens, sigma=2.0)
    assert segmented.max() == 1
    assert segmented.min() == 0
    assert np.sum(segmented) > 0


def test_segment_combined_with_no_objects():
    # Create images with no objects
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    segmented = segment_combined(image_SE2, image_inlens)
    assert segmented.max() == 0
    assert segmented.min() == 0
    assert np.sum(segmented) == 0


# Test cases for combine_images


def test_combine_images_basic():
    # Create simple binary images
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[41:61, 41:61] = 255
    combined = combine_images(image_SE2, image_inlens)
    assert combined.shape == (100, 100)
    assert combined.dtype == np.uint8
    assert (combined[40:60, 40] == 127).all()
    assert (combined[60, 41:61] == 127).all()
    assert (combined[41:60, 41:60] == 254).all()
    assert combined.max() == 254
    assert combined.min() == 0


def test_combine_images_with_noise():
    # Create images with noise
    np.random.seed(0)
    image_SE2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    image_inlens = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    combined = combine_images(image_SE2, image_inlens)
    assert combined.shape == (100, 100)
    assert combined.dtype == np.uint8


def test_combine_images_with_different_shapes():
    # Create images with different shapes
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        combine_images(image_SE2, image_inlens)


def test_combine_images_with_all_zeros():
    # Create images with all zeros
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    combined = combine_images(image_SE2, image_inlens)
    assert combined.shape == (100, 100)
    assert combined.max() == 0
    assert combined.min() == 0


def test_combine_images_with_all_white():
    # Create images with all white pixels
    image_SE2 = np.ones((100, 100), dtype=np.uint8) * 255
    image_inlens = np.ones((100, 100), dtype=np.uint8) * 255
    combined = combine_images(image_SE2, image_inlens)
    assert combined.shape == (100, 100)
    assert combined.max() == 254
    assert combined.min() == 254


def test_combine_images_with_all_ones():
    # Create images with all ones (demonstate rounding error from integer averaging)
    image_SE2 = np.ones((100, 100), dtype=np.uint8)
    image_inlens = np.ones((100, 100), dtype=np.uint8)
    combined = combine_images(image_SE2, image_inlens)
    assert combined.shape == (100, 100)
    assert combined.max() == 0
    assert combined.min() == 0


# Test cases for combine_segmentation_with_overlay


def test_combine_segmentation_with_overlay_basic():
    # Create simple binary images and a segmentation mask
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    segmented = np.zeros((100, 100), dtype=np.uint8)
    image_SE2[40:60, 40:60] = 255
    image_inlens[41:61, 41:61] = 255
    segmented[45:55, 45:55] = 1
    result = combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8
    assert (result[45:55, 45:55, 0] == 255).all()
    assert (result[45:55, 45:55, 1] == 255).all()
    assert (result[45:55, 45:55, 2] == 0).all()


def test_combine_segmentation_with_overlay_with_noise():
    # Create images with noise and a segmentation mask
    np.random.seed(0)
    image_SE2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    image_inlens = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    segmented = np.zeros((100, 100), dtype=np.uint8)
    segmented[45:55, 45:55] = 1
    result = combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8
    assert (result[45:55, 45:55, 0] == 255).all()
    assert (result[45:55, 45:55, 1] == 255).all()
    assert (result[45:55, 45:55, 2] == 0).all()


def test_combine_segmentation_with_overlay_with_different_shapes():
    # Create images with different shapes and a segmentation mask
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((50, 50), dtype=np.uint8)
    segmented = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)


def test_combine_segmentation_with_overlay_with_all_zeros():
    # Create images with all zeros and an empty segmentation mask
    image_SE2 = np.zeros((100, 100), dtype=np.uint8)
    image_inlens = np.zeros((100, 100), dtype=np.uint8)
    segmented = np.zeros((100, 100), dtype=np.uint8)
    result = combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)
    assert result.shape == (100, 100, 3)
    assert result.max() == 0
    assert result.min() == 0


def test_combine_segmentation_with_overlay_with_all_white():
    # Create images with all white pixels and a segmentation mask
    image_SE2 = np.ones((100, 100), dtype=np.uint8) * 255
    image_inlens = np.ones((100, 100), dtype=np.uint8) * 255
    segmented = np.zeros((100, 100), dtype=np.uint8)
    segmented[45:55, 45:55] = 1
    result = combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)
    assert result.shape == (100, 100, 3)
    assert (result[45:55, 45:55, 0] == 255).all()
    assert (result[45:55, 45:55, 1] == 255).all()
    assert (result[45:55, 45:55, 2] == 0).all()
    assert result.max() == 255
    assert result[..., 0:1].min() == 254


def test_combine_segmentation_with_overlay_with_all_ones():
    # Create images with all ones and a segmentation mask
    image_SE2 = np.ones((100, 100), dtype=np.uint8)
    image_inlens = np.ones((100, 100), dtype=np.uint8)
    segmented = np.zeros((100, 100), dtype=np.uint8)
    segmented[45:55, 45:55] = 1
    result = combine_segmentation_with_overlay(image_SE2, image_inlens, segmented)
    assert result.shape == (100, 100, 3)
    assert (result[45:55, 45:55, 0] == 255).all()
    assert (result[45:55, 45:55, 1] == 255).all()
    assert (result[45:55, 45:55, 2] == 0).all()
    assert (result[35:44, 35:44, :] == 0).all()
    assert result.max() == 255
    assert result.min() == 0


# Test cases for evaluate_segmentation


def test_evaluate_segmentation_basic():
    # Create a simple label image with a single object
    label_image = np.zeros((100, 100), dtype=np.uint8)
    label_image[40:50, 40:60] = 1
    df = evaluate_segmentation(label_image)
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "area" in df.columns
    assert "axis_major_length" in df.columns
    assert "axis_minor_length" in df.columns
    assert "centroid-0" in df.columns
    assert "centroid-1" in df.columns
    assert "orientation" in df.columns
    assert df.shape[0] == 1
    assert df["area"].iloc[0] == 10 * 20 * pixel_size_um**2
    assert df["axis_minor_length"].iloc[0] == pytest.approx(0.06382847, rel=1e-6)
    assert df["axis_major_length"].iloc[0] == pytest.approx(0.12813958, rel=1e-6)
    assert df["centroid-0"].iloc[0] == pytest.approx(44.5 * pixel_size_um, rel=1e-6)
    assert df["centroid-1"].iloc[0] == pytest.approx(49.5 * pixel_size_um, rel=1e-6)
    # should be either +90 or -90 degrees in radians
    assert np.abs(df["orientation"].iloc[0]) == pytest.approx(1.570796, rel=1e-6)


def test_evaluate_segmentation_multiple_objects():
    # Create a label image with multiple objects
    label_image = np.zeros((100, 100), dtype=np.uint8)
    label_image[10:30, 10:30] = 1
    label_image[70:90, 70:90] = 2
    df = evaluate_segmentation(label_image)
    assert df.shape[0] == 2
    assert set(df["label"]) == {1, 2}


def test_evaluate_segmentation_no_objects():
    # Create a label image with no objects
    label_image = np.zeros((100, 100), dtype=np.uint8)
    df = evaluate_segmentation(label_image)
    assert df.shape[0] == 0


def test_evaluate_segmentation_with_noise():
    # Create a label image with noise and a single object
    np.random.seed(0)
    label_image = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    label_image[40:60, 40:60] = 1
    df = evaluate_segmentation(label_image)
    assert df.shape[0] > 0


def test_evaluate_segmentation_small_image():
    # Create a small label image with a single object
    label_image = np.zeros((20, 20), dtype=np.uint8)
    label_image[10:15, 10:15] = 1
    df = evaluate_segmentation(label_image)
    assert df.shape[0] == 1


def test_evaluate_segmentation_custom_properties():
    # Create a simple label image with a single object
    label_image = np.zeros((100, 100), dtype=np.uint8)
    label_image[40:60, 40:60] = 1
    properties = ("label", "area", "perimeter")
    df = evaluate_segmentation(label_image, properties)
    assert "perimeter" in df.columns
    assert df.shape[0] == 1


# Test cases for process_folder


@patch("carde.image_processing.Path.mkdir")
@patch("carde.image_processing.imsave")
@patch("carde.image_processing.read_image_pair")
@patch("carde.image_processing.Path.glob")
@patch("carde.image_processing.natsorted")
@patch("carde.image_processing.pd.DataFrame.to_csv")
def test_process_folder(
    mock_to_csv,
    mock_natsorted,
    mock_glob,
    mock_read_image_pair,
    mock_imsave,
    mock_mkdir,
):
    # Setup mock paths
    mock_image_path1 = MagicMock(spec=Path)
    mock_image_path1.name = "image_01.tif"
    mock_image_path1.stem = "image_01"
    mock_image_path2 = MagicMock(spec=Path)
    mock_image_path2.name = "image_02.tif"
    mock_image_path2.stem = "image_02"
    mock_glob.return_value = [mock_image_path1, mock_image_path2]
    mock_natsorted.return_value = [mock_image_path1, mock_image_path2]

    # Setup mock return values
    image1 = np.zeros((100, 100), dtype=np.uint8)
    image1[40:60, 40:60] = 255
    image1[90:, :] = 255
    image2 = np.zeros((100, 100), dtype=np.uint8)
    image2[40:60, 40:60] = 255
    image2[90:, :] = 255
    mock_read_image_pair.return_value = (image1, image2)

    expected_segmentation = np.zeros((89, 100), dtype=np.uint8)
    expected_segmentation[40:60, 40:60] = 1

    # Call the function
    process_folder(Path("/mock/image_folder"), Path("/mock/output_folder"))

    # Assertions
    mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
    mock_glob.assert_called_once_with("*.tif")
    mock_natsorted.assert_called_once_with([mock_image_path1, mock_image_path2])
    assert mock_read_image_pair.call_count == 1  # because the matching image_02 should be skipped
    assert mock_imsave.call_count == 2
    assert mock_to_csv.call_count == 1
    # assert mock_to_csv.call_count == 1
    args, kwargs = mock_imsave.call_args_list[0]
    assert args[0] == Path("/mock/output_folder/image_01-02_label.tif")
    np.testing.assert_array_equal(args[1], expected_segmentation)
    args, kwargs = mock_imsave.call_args_list[1]
    assert args[0] == Path("/mock/output_folder/image_01-02_overlay.bmp")
    assert args[1].shape == (100, 100, 3)
    assert args[1].dtype == np.uint8
    # assert that the segmentation is there and has the correct values
    assert (args[1][40:60, 40:60, 0] == 255).all()
    assert (args[1][40:60, 40:60, 1] == 255).all()
    assert (args[1][40:60, 40:60, 2] == 0).all()
    # assert that the metadata is there and has the correct value
    assert (args[1][90:, :, :] == 254).all()
    args, kwargs = mock_to_csv.call_args_list[0]
    assert args[0] == Path("/mock/output_folder/image_01-02_table.csv")
