import numpy as np
from PIL import Image as PILImage


def segmentLines(img: PILImage.Image, min_gap=8):
    '''
    Segment the images into lines so that it becomes easier to read

    :param img:         Actual Image
    :param min_gap:     Gap before we consider it the end of the line
    :return:            Array of each line of text
    '''

    gray = np.array(img.convert("L")) #Makes the image grayscale for easier understanding

    # We get the high and low brightness points to distinguish the Ink and the Paper.
    # It's called the Otsu's method according to my research
    flat = gray.flatten()
    low, high = np.percentile(flat, [1, 95])
    binarize_threshold = (low + high) / 2


    dark_pixels = gray < binarize_threshold
    dark_count_per_row = dark_pixels.sum(axis=1)
    print(f"Dark pixel counts per row - min: {dark_count_per_row.min()}, max: {dark_count_per_row.max()}, mean: {dark_count_per_row.mean()}")

    min_dark_pixels = gray.shape[1] * 0.01
    print(f"min_dark_pixels threshold: {min_dark_pixels}")

    is_text_row = dark_count_per_row > min_dark_pixels
    pattern = "".join("T" if t else "." for t in is_text_row)
    print(pattern)

    lines = []
    start = None
    gap_count = 0
    for i, has_text in enumerate(is_text_row):
        if has_text:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap:
                    lines.append((start, i - gap_count))
                    start = None
    if start is not None:
        lines.append((start, len(is_text_row)))
    padding = 10
    return [
        img.crop((0, max(0, top - padding), img.width, min(img.height, bottom + padding)))
        for top, bottom in lines
    ]