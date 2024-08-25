# Image Cropper

A tool to automatically crop images for a humanoid subject to the closest aspect ratio from a defined list. You can also crop to just the face rather than the whole subject.

This can be used to crop images to supported aspect ratios for image generation AI models (e.g. SDXL).

Note that it does NOT crop to supported _resolutions_.
The expectation is that the training tool will resize the image to a supported resolution.

## Usage

```
usage: crop.py [-h] --out OUT [--limit LIMIT] [--padding PADDING] [--border BORDER] [--debug] [--force] [--copy-small]
               [--sdxl] [--seg-class {0,1,2,3,4,5}] [--min-edge MIN_EDGE]
               inputs [inputs ...]

Process images in a folder.

positional arguments:
  inputs                Source files / dir

options:
  -h, --help            show this help message and exit
  --out OUT             The output directory
  --limit LIMIT         Limit the number of files to crop (default: None)
  --padding PADDING     Subject padding px or fractional percent (default: 40)
  --border BORDER       Number of px to remove from image border (default: 0)
  --debug               Debug mode (default: False)
  --force               Overwrite existing file if present (default: False)
  --copy-small          Copy images that dont meet min-edge (default: False)
  --sdxl                Use SDXL aspect ratios (default: False)
  --seg-class {0,1,2,3,4,5}
                        Segmentation class (default: None)
  --min-edge MIN_EDGE   Skip images with an edge < minimum (default: 1024)
```

### Segmentation Classes

By default, the tool will crop to the subject but other segmentation options are possible:

| Args | Class                |
|------|----------------------|
| 0    | background           |
| 1    | hair                 |
| 2    | body-skin            |
| 3    | face-skin            |
| 4    | clothes              |
| 5    | others (accessories) |

Multiple options can be used together.

For face cropping, use either 3 & 1 or 3 an increase the padding.

### Debug mode

Debug mode prints more information to the console and outputs the segmentation mask to the output directory.

# Setup

#### Create a venv

```
python -m venv .\venv
```

#### Activate the venv

```
.\venv\Scripts\activate
```

Use `Activate.ps1` or `activate.bat` for Windows

#### Install dependencies

```
pip install -r .\requirements.txt
```

#### Execution

```
python crop.py <ARGS>
```

# EXE Build

The tool can be packaged as an EXE. This makes it easier to execute from a windows process.

#### Install pyinstaller

```
pip install pyinstaller
```

You can do this outside of the venv

#### Package the tool

```
pyinstaller .\crop.spec --noconfirm
```

You should now have an executable in the build dir.