import os
from os.path import exists,isdir,isfile,join
from PIL import Image
from retinaface import RetinaFace
from concurrent.futures import ProcessPoolExecutor

Image.MAX_IMAGE_PIXELS = 200000000  # Allow images up to 200 million pixels

# width / height
RESOLUTIONS = [
    (768, 1344),
    (1,1),
    (1216,832)
]

NUM_THREADS = 1  # Modify this for the desired number of threads
ROOT = r"S:\pictures\"
#OUTPUT_ROOT = r"S:\pictures\"  # Root folder to save resized images

def resize_image(args):
    input_path, resolution, output_root = args

    sets = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    cropped = os.join(sets,"Cropped")

    for c in cropped:
        print(f"{c} - {resolution}")

    # Load the image
    # image = Image.open(input_path)
    
    # # Get face locations using RetinaFace
    # faces = RetinaFace.detect_faces(input_path)
    # face_locations = None

    # if faces and 'face_1' in faces:
    #     face = faces['face_1']
    #     x, y, x2, y2 = face['facial_area']
    #     face_locations = [(y, x2, y2, x)]  # Adjust format to match face_recognition
    
    # desired_aspect_ratio = resolution[0] / resolution[1]
    # image_aspect_ratio = image.width / image.height

    # # Calculate crop dimensions
    # if image_aspect_ratio > desired_aspect_ratio:
    #     new_width = int(image.height * desired_aspect_ratio)
    #     new_height = image.height
    # else:
    #     new_width = image.width
    #     new_height = int(image.width / desired_aspect_ratio)

    # # Default centering
    # left = (image.width - new_width) / 2
    # top = (image.height - new_height) / 2
    # right = (image.width + new_width) / 2
    # bottom = (image.height + new_height) / 2

    # # Adjust for face center if a face is detected
    # if face_locations:
    #     face_top, face_right, face_bottom, face_left = face_locations[0]
    #     face_center_x = (face_left + face_right) // 2
    #     face_center_y = (face_top + face_bottom) // 2

    #     left = min(max(0, face_center_x - new_width // 2), image.width - new_width)
    #     top = min(max(0, face_center_y - new_height // 2), image.height - new_height)
    #     right = left + new_width
    #     bottom = top + new_height

    # image = image.crop((left, top, right, bottom))

    # # Resize image with best resampling filter (LANCZOS)
    # #image = image.resize(resolution, Image.LANCZOS)
    
    # output_folder = os.path.join(output_root, f"{resolution[0]}x{resolution[1]}")
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # output_path = os.path.join(output_folder, os.path.basename(input_path))
    # image.save(output_path, quality=100)

def crop_image(input_path, resolution, output_path,buffer, image):
    
    if image is None:
        image = Image.open(input_path)

    # Get face locations using RetinaFace
    faces = RetinaFace.detect_faces(input_path)
    face_locations = None

    if faces and 'face_1' in faces:
        face = faces['face_1']
        x, y, x2, y2 = face['facial_area']
        face_locations = [(y, x2, y2, x)]  # Adjust format to match face_recognition
    
    desired_aspect_ratio = resolution[0] / resolution[1]
    image_aspect_ratio = image.width / image.height

    # Calculate crop dimensions
    if image_aspect_ratio > desired_aspect_ratio:
        new_width = int(image.height * desired_aspect_ratio)
        new_height = image.height
    else:
        new_width = image.width
        new_height = int(image.width / desired_aspect_ratio)

    # Default centering
    left = (image.width - new_width) / 2
    top = (image.height - new_height) / 2
    right = (image.width + new_width) / 2
    bottom = (image.height + new_height) / 2

    # Adjust for face center if a face is detected
    if face_locations:
        face_top, face_right, face_bottom, face_left = face_locations[0]
        face_center_x = (face_left + face_right) // 2
        face_center_y = (face_top + face_bottom) // 2

        left = min(max(0, face_center_x - new_width // 2), image.width - new_width)
        top = min(max(0, face_center_y - new_height // 2), image.height - new_height)
        right = left + new_width
        bottom = top + new_height

    if buffer:
        print(f"buffer: {buffer}")
        left = left + buffer
        right = right - buffer
        top = top + buffer
        bottom = bottom - buffer


    image = image.crop((left, top, right, bottom))

    # output_folder = join(output_root, f"{resolution[0]}x{resolution[1]}")
    # if not exists(output_folder):
    #     os.makedirs(output_folder)

    #output_folder = os.path.dirname(os.path.dirname(input_path))

    #output_path = join(output_folder, os.path.basename(input_path))
    print(f"save to {output_path}")
    image.save(output_path, quality=100)

def closest_resolution(image):
    image_aspect_ratio = image.width / image.height
    best_diff = 99
    best_res = None
    for res in RESOLUTIONS:
        res_aspect_ratio = res[0] / res[1]
        diff = abs(res_aspect_ratio - image_aspect_ratio)
        if (diff < best_diff):
            best_diff = diff
            best_res = res
    
    return best_res


def process_set(set):
    name,dir = set
    print(f"{name} - {dir}")
    todo_dir = join(dir,"todo")
    image_paths = [join(todo_dir, fname) for fname in os.listdir(todo_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    total_images = len(image_paths)
    processed_count = 0

    for image_path in image_paths:
        processed_count += 1
        skip = exists(join(dir,os.path.basename(image_path)))
        if skip:
            print(f"skipping {os.path.basename(image_path)}")
            #image = Image.open(image_path)
            #res = closest_resolution(image)
            #crop_image(image_path, res, image_path, image)
            #print(f"{image_path}")
            #print(f"Processed {processed_count}/{total_images} images")
        else:
            image = Image.open(image_path)
            res = closest_resolution(image)
            crop_image(image_path, res, join(dir, os.path.basename(image_path)), 40, image)
            print(f"Processed {processed_count}/{total_images} images")


    # with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
    #     for _ in executor.map(crop_image, [(path, resolution, ROOT) for path in image_paths]):
    #         processed_count += 1
    #         print(f"Processed {processed_count}/{total_images} images for resolution {resolution[0]}x{resolution[1]}...")

if __name__ == "__main__":
    sets = [ (f, join(ROOT,f)) for f in os.listdir(ROOT) if isdir(join(ROOT, f)) ]#[:1]
    todo_sets = filter(lambda set: exists(join(set[1],"todo")), sets)

    for set in todo_sets:
        name,dir = set
        process_set(set)
        # if (name == "Twistys - Clean and Warm"):
        #     process_set(set)
    print("Processing complete!")
