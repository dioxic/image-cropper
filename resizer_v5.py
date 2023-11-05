import argparse
import os
from os.path import exists, isdir, isfile, join
import sys
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import math
from functools import reduce
from typing import Tuple, Union

Image.MAX_IMAGE_PIXELS = 200000000  # Allow images up to 200 million pixels

# width / height
RESOLUTIONS = [
    (9,16),
    #(832, 1216),
    (896, 1152),
    (1024, 1024),
    (1152, 896),
    #(1216, 832),
    (16,9),
]

SDXL_RESOLUTIONS = [
    (640, 1536),
    (768, 1344),
    (832, 1216),
    (896, 1152),
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1536, 640)
]

DEBUG = True
NUM_THREADS = 1  # Modify this for the desired number of threads


def debug(msg):
    if DEBUG:
        print(msg)

def get_resource_dir():
    return getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

def highestArea(a,b):
    area_a = a.bounding_box.width * a.bounding_box.height
    area_b = b.bounding_box.width * b.bounding_box.height
    if area_a > area_b:
        return a
    else:
        return b
    
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px
    
def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
    Returns:
    Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                            width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

def get_face(input_image, output_mask_path):
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.abspath(os.path.join(get_resource_dir(), 'blaze_face_short_range.tflite'))),
        running_mode=mp.tasks.vision.RunningMode.IMAGE)
    
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))

        # Perform face detection on the provided single image.
        # The face detector must be created with the image mode.
        detector_result = detector.detect(image)

        if len(detector_result.detections) == 0:
            return None

        biggest_face = reduce(highestArea, detector_result.detections).bounding_box

        if output_mask_path and DEBUG:
            image_copy = np.copy(image.numpy_view())
            annotated_image = visualize(image_copy, detector_result)
            rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            debug(f"saving bounded box to {output_mask_path}")
            cv2.imwrite(output_mask_path, rgb_annotated_image)

        return [
            biggest_face.origin_x,
            biggest_face.origin_y,
            biggest_face.origin_x + biggest_face.width,
            biggest_face.origin_y + biggest_face.height
        ]

# classes:
# 0 - background
# 1 - hair
# 2 - body-skin
# 3 - face-skin
# 4 - clothes
# 5 - others (accessories)
def get_subject(input_image, output_mask_path, classes):
    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white

    if classes is None:
        model = 'selfie_segmenter.tflite'
    else:
        model = 'selfie_multiclass.tflite'
        
    # Create the options that will be used for ImageSegmenter
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.abspath(os.path.join(get_resource_dir(), model))),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True)

    # Create the image segmenter
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))
        image_data = image.numpy_view()

        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        print(segmentation_result)
        print(f"length: {len(segmentation_result.confidence_masks)}")

        if classes is None:
            mask = segmentation_result.confidence_masks[0].numpy_view()
        else:
            mask = np.zeros(image_data.shape, dtype=np.uint8)[:,:,0]
            for classIdx in classes:
                mask = np.maximum(mask, segmentation_result.confidence_masks[classIdx].numpy_view())

        # Generate solid color images for showing the output segmentation mask.
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        # Stack the segmentation mask for 3 RGB channels, and then create a filter for which pixels to keep
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        output_image = np.where(condition, fg_image, bg_image)
        subject = np.argwhere(output_image == MASK_COLOR)

        if len(subject) == 0:
            return None
        
        print(subject)

        y1, x1, z1 = subject.min(axis=0)
        y2, x2, z2 = subject.max(axis=0)

        debug(f"output mask: x1:{x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        if output_mask_path and DEBUG:
            debug(f"saving mask to {output_mask_path}")
            cv2.imwrite(output_mask_path, output_image)
        return [x1, y1, x2, y2]

def crop_image(image, subject, resolutions, padding, border=0):
    subject_left, subject_top, subject_right, subject_bottom = subject
    subject_height = subject_bottom - subject_top
    subject_width = subject_right - subject_left

    debug(f"subject: {subject}, width: {subject_width}, height: {subject_height}")

    if padding:
        if padding < 1:
            # assume %
            padding_width = min(subject_height, subject_width) * padding
            padding_height = min(subject_height, subject_width) * padding
        else:
            padding_width = padding
            padding_height = padding

        subject_left = max(subject_left - padding_width, border)
        subject_right = min(subject_right + padding_width, image.width - border)
        subject_top = max(subject_top - padding_height, border)
        subject_bottom = min(subject_bottom + padding_height, image.height - border)
        subject_height = subject_bottom - subject_top
        subject_width = subject_right - subject_left
        subject = [subject_left, subject_top, subject_right, subject_bottom]
        debug(f"padded subject: {subject}, width: {subject_width}, height: {subject_height}")

    new_width, new_height = best_resolution(subject, image, resolutions, border)

    # centre on subject
    subject_center_x = (subject_left + subject_right) // 2
    subject_center_y = (subject_top + subject_bottom) // 2

    left = min(max(0, subject_center_x - new_width // 2), image.width - new_width)
    top = min(max(0, subject_center_y - new_height // 2), image.height - new_height)
    right = left + new_width
    bottom = top + new_height

    if border:
        if new_width > (image.width - (border * 2)):
            print(f"border: {border}, img width: {image.width}, target width: {new_width}")
            raise "error not enough room for border (width)"
        elif new_height > (image.height - (border * 2)):
            print(f"border: {border}, img height: {image.height}, target height: {new_height}")
            raise "error not enough room for border (height)"

        if left < border:
            diff = border - left
            left = left + diff
            right = right + diff
        if right > image.width - border:
            diff = right - (image.width - border)
            right = right - diff
            left = left - diff
        if top < border:
            diff = border - top
            top = top + diff
            bottom = bottom + diff
        if bottom > image.height - border:
            diff = bottom - (image.height - border)
            bottom = bottom - diff
            top = top - diff

    debug(f"cropping to: left:{left}, top:{top}, right:{right}, bottom:{bottom}")
    return image.crop((left, top, right, bottom))


def best_resolution(subject, image, resolutions, border):
    subject_left, subject_top, subject_right, subject_bottom = subject
    subject_height = subject_bottom - subject_top
    subject_width = subject_right - subject_left
    subject_aspect_ratio = subject_width / subject_height
    best_diff = 4096*4096
    best_dimensions = None
    best_fits = False

    for res in resolutions:
        res_aspect_ratio = res[0] / res[1]
        debug("------------------")
        debug(f"  resolution: {res} - aspect: {res_aspect_ratio}")
        new_width, new_height = apply_resolution(subject, image, res, border)

        diff = int(abs((new_width*new_height) - (subject_width*subject_height)))
        fits = new_width >= subject_width-4 and new_height >= subject_height-4

        debug(f"  fits: {fits}, width: {new_width}, height: {new_height}, diff: {diff}")

        if (fits and diff < best_diff) or (fits and not best_fits):
            best_dimensions = [new_width, new_height]
            best_fits = True
            best_diff = diff
        elif not best_fits and diff < best_diff:
            best_dimensions = [new_width, new_height]
            best_diff = diff

    debug("------------------")
    debug(f"best resolution to {subject_aspect_ratio}: {best_dimensions}")
    return best_dimensions


def apply_resolution(subject, image, resolution, border=0):
    subject_left, subject_top, subject_right, subject_bottom = subject
    subject_height = subject_bottom - subject_top
    subject_width = subject_right - subject_left
    image_aspect_ratio = image.width / image.height
    res_aspect_ratio = resolution[0] / resolution[1]

    # going more horizontal
    if image_aspect_ratio > res_aspect_ratio:
        max_height = image.height - (border * 2)
        max_width = int(max_height * res_aspect_ratio)
    else:
        max_width = image.width - (border * 2)
        max_height = int(max_width / res_aspect_ratio)

    debug(f"  max height: {max_height}, width: {max_width}")

    new_height = int(max_height / 4)
    new_width = int(new_height * res_aspect_ratio)

    while (new_width < subject_width or new_height < subject_height) and new_height < max_height and new_width < max_width:
        # print(f"new width: {new_width}, new height: {new_height}")
        new_height += 1
        new_width = int(new_height * res_aspect_ratio)

    return [new_width, new_height]


def process_images(image_paths, output_path, limit, padding, border, force, resolutions, classes):
    processed_count = 0
    image_count = len(image_paths)

    for image_path in image_paths:
        target = join(output_path, os.path.basename(image_path))
        if not force and exists(target):
            debug(f"skipping {os.path.basename(image_path)} - file exists")
            image_count -= 1
        else:
            image = Image.open(image_path)
            output_mask_path = join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}-debug.jpg")
            subject = get_subject(image, output_mask_path, classes)

            if subject is not None:
                cropped = crop_image(image, subject, resolutions, padding, border)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                cropped.save(target, quality=100)
                processed_count += 1
            else:
                print(f"skipping {os.path.basename(image_path)} - cannot find subject")
                image_count -= 1
            print(f"Processed {processed_count}/{image_count} images")

        if processed_count == limit:
            break

    return processed_count


def main():
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    parser.add_argument('files', nargs="+", help="Source files")
    parser.add_argument('--out', required=True, help='The output directory.')
    parser.add_argument('--limit', required=False, help='Limit the number of files to crop.', type=int)
    parser.add_argument('--padding', default=40, help='Subject padding px or fractional %.', type=float)
    parser.add_argument('--border', default=0, help='Number of px to remove from image border.', type=int)
    parser.add_argument('--debug', help='Debug mode.', action='store_true', default=False)
    parser.add_argument('--force', help='Overwrite existing file if present.', action='store_true', default=False)
    parser.add_argument('--sdxl', help='Use SDXL aspect ratios.', action='store_true', default=False)
    #parser.add_argument('--face', help='Crop to face.', action='store_true', default=False)
    parser.add_argument('--seg', help='Segmentation class.', action='append', type=int)

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    if args.sdxl:
        res = SDXL_RESOLUTIONS
    else:
        res = RESOLUTIONS

    process_images(args.files, args.out, args.limit, args.padding, args.border, args.force, res, args.seg)

    print("Processing complete!")


if __name__ == "__main__":
    main()
