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
    # It is approximate to Otsu's method which gets an auto-threshold.
    flat = gray.flatten()
    low, high = np.percentile(flat, [1, 95])
    binarize_threshold = (low + high) / 2

    # We compare each number in gray to see if it is ink or if it is blank
    dark_pixels = gray < binarize_threshold

    #At this point darkPixels are an array of true and false. This sums them up to see how many dark pixels the row has
    dark_count_per_row = dark_pixels.sum(axis=1)

    #Gets 1% of the amount of pixels the row can have.
    min_dark_pixels = gray.shape[1] * 0.01

    #Checks if the row could have text.
    is_text_row = dark_count_per_row > min_dark_pixels

    lines = []
    start = None
    gap_count = 0

    for i, has_text in enumerate(is_text_row):
        if has_text:
            #If it hasn't started then it starts a new "line"
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                # If it started then it starts counting the line in-between text and it appends it to count it as a line
                gap_count += 1
                if gap_count >= min_gap:
                    lines.append((start, i - gap_count))
                    start = None

    if start is not None:
        lines.append((start, len(is_text_row)))
    # Similar to a flush, incase we leave the loop and we still haven't finished the line.



    padding = 10
    #Sends back the original image (not grayscale) with the text data we got from the grayscale image.
    #Also adds 10 pixels of padding to make sure it is not cutting off the text.
    return [
        img.crop((0, max(0, top - padding), img.width, min(img.height, bottom + padding)))
        for top, bottom in lines
    ]