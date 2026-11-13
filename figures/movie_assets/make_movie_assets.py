"""Cut the fly and VR-rig schematics used in the supplemental movies out of the Figure 1 artwork.

Run ``python make_movie_assets.py [path/to/Fig1_PresynapticGating.png]``. The schematics are the top-down fly of panel c
(the upright one, without its turning arrow) and the two-photon / VR rig of panel e. The white background around each is
made transparent by flood-filling from the image border, so white enclosed by outlines (the fly's wings) stays opaque.
"""

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).parent
DEFAULT_SOURCE = pathlib.Path('/home/mplitt/Downloads/Fig1_PresynapticGating.png')
# (left, upper, right, lower) in pixels of the 3921 x 2474 figure
CROPS = {'fly_topdown.png': (1685, 325, 1920, 540),
         'vr_rig.png': (210, 1240, 565, 1605)}
SENTINEL = (255, 0, 255)                     # a color absent from the artwork, marking the background to remove


def transparent_background(image, thresh=40):
    """RGBA copy of `image` with the white region connected to its border made transparent."""
    rgb = image.convert('RGB')
    w, h = rgb.size
    for x in range(w):
        for y in (0, h - 1):
            if rgb.getpixel((x, y)) != SENTINEL:
                ImageDraw.floodfill(rgb, (x, y), SENTINEL, thresh=thresh)
    for y in range(h):
        for x in (0, w - 1):
            if rgb.getpixel((x, y)) != SENTINEL:
                ImageDraw.floodfill(rgb, (x, y), SENTINEL, thresh=thresh)
    rgba = np.array(rgb.convert('RGBA'))
    rgba[(rgba[..., :3] == SENTINEL).all(axis=-1)] = 0
    return Image.fromarray(rgba)


def main(source=DEFAULT_SOURCE):
    figure = Image.open(source)
    for name, box in CROPS.items():
        transparent_background(figure.crop(box)).save(HERE / name)
        print('wrote', HERE / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', nargs='?', type=pathlib.Path, default=DEFAULT_SOURCE)
    main(parser.parse_args().source)
