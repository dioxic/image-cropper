import argparse
import os
from os.path import exists, isdir, isfile, join
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from concurrent.futures import ProcessPoolExecutor

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


# OUTPUT_ROOT = r"S:\pictures\Panna\MediaMynx\Cropped\018"  # Root folder to save resized images

def debug(msg):
    if DEBUG:
        print(msg)


def get_subject(input_image, output_mask_path):
    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white

    # Create the options that will be used for ImageSegmenter
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='selfie_segmenter.tflite'),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True)

    # Create the image segmenter
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))

        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        # Generate solid color images for showing the output segmentation mask.
        image_data = image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        subject = np.argwhere(output_image == BG_COLOR)

        if len(subject) == 0:
            return None

        y1, x1, z1 = subject.min(axis=0)
        y2, x2, z2 = subject.max(axis=0)

        debug(f"output mask: x1:{x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        if output_mask_path and DEBUG:
            debug(f"saving mask to {output_mask_path}")
            cv2.imwrite(output_mask_path, output_image)
        return [x1, y1, x2, y2]


def detect_pose(image_path, output_dir):
    # utils for drawing on image
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # mediapipe pose model
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    # convert image to RGB (just for input to model)
    image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get results using mediapipe
    results = mp_pose.process(image_input)
    x1 = 1
    y1 = 1
    x2 = 0
    y2 = 0

    for idx in range(len(results.pose_landmarks.landmark)):
        lm = results.pose_landmarks.landmark[idx]

        if lm.visibility > 0.5:
            # print(f"{lm.visibility}")
            x1 = max(min(x1, lm.x), 0)
            x2 = min(max(x2, lm.x), 1)
            y1 = max(min(y1, lm.y), 0)
            y2 = min(max(y2, lm.y), 1)

    if not results.pose_landmarks:
        print("no results found")
    else:
        image_name, extension = os.path.splitext(os.path.basename(image_path))
        print(f"{image_name} - min x:{x1}, max x: {x2}, min y: {y1}, max y: {y2}")
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # write image to storage
        cv2.imwrite(join(output_dir, f"{image_name}-processed.{extension}"), image)


def crop_image(image, subject, resolutions, padding, border=0):
    subject_left, subject_top, subject_right, subject_bottom = subject
    subject_height = subject_bottom - subject_top
    subject_width = subject_right - subject_left

    debug(f"subject: {subject}, width: {subject_width}, height: {subject_height}")

    if padding:
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

    if image_aspect_ratio > res_aspect_ratio:
        max_height = image.height - (border * 2)
        max_width = int(max_height * res_aspect_ratio)
    else:
        max_width = image.width - (border * 2)
        max_height = int(max_width / res_aspect_ratio)

    debug(f"  max height: {max_height}, width: {max_width}")

    new_height = int(max_height / 2)
    new_width = int(new_height * res_aspect_ratio)

    while (new_width < subject_width or new_height < subject_height) and new_height < max_height and new_width < max_width:
        # print(f"new width: {new_width}, new height: {new_height}")
        new_height += 1
        new_width = int(new_height * res_aspect_ratio)

    return [new_width, new_height]


def closest_resolution(aspect_ratio, resolutions):
    best_diff = 99
    best_res = None
    for res in resolutions:
        res_aspect_ratio = res[0] / res[1]
        diff = abs(res_aspect_ratio - aspect_ratio)
        if diff < best_diff:
            best_diff = diff
            best_res = res

    debug(f"closest resolution to {aspect_ratio}: {best_res}")
    return best_res


def process_set(set, limit, padding, border, force, resolutions):
    name, dir = set
    print(f"{name} - {dir}")
    todo_dir = join(dir, "todo")
    image_paths = [join(todo_dir, fname) for fname in os.listdir(todo_dir) if
                   fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    processed_count = 0
    image_count = len(image_paths)

    for image_path in image_paths:
        if not force and exists(join(dir, os.path.basename(image_path))):
            debug(f"skipping {os.path.basename(image_path)} - file exists")
            image_count -= 1
        else:
            image = Image.open(image_path)
            output_mask_path = join(dir, f"{os.path.splitext(os.path.basename(image_path))[0]}-processed.jpg")
            subject = get_subject(image, output_mask_path)
            if subject is not None:
                cropped = crop_image(image, subject, resolutions, padding, border)
                cropped.save(join(dir, os.path.basename(image_path)), quality=100)
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
    parser.add_argument('--dir', required=True, help='The source folder.')
    parser.add_argument('--limit', required=False, help='Limit the number of files to crop.', type=int)
    parser.add_argument('--padding', required=False, help='Number of px to pad the subject.', type=int)
    parser.add_argument('--border', default=0, help='Number of px to remove from image border.', type=int)
    parser.add_argument('--debug', help='Debug mode.', action='store_true', default=False)
    parser.add_argument('--sets', help='Set folders to process.')
    parser.add_argument('--force', help='Overwrite existing file if present.', action='store_true', default=False)
    parser.add_argument('--sdxl', help='Use SDXL aspect ratios.', action='store_true', default=False)

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    if args.sdxl:
        res = SDXL_RESOLUTIONS
    else:
        res = RESOLUTIONS

    sets = [(f, join(args.dir, f)) for f in os.listdir(args.dir) if isdir(join(args.dir, f))]  # [:1]
    todo_sets = filter(lambda todo: exists(join(todo[1], "todo")), sets)
    for todo_set in todo_sets:
        name = todo_set[0]
        if args.sets == name or args.sets is None:
            processed_img_count = process_set(todo_set, args.limit, args.padding, args.border, args.force, res)
            if args.limit is not None and processed_img_count >= args.limit:
                break
    print("Processing complete!")


if __name__ == "__main__":
    main()
