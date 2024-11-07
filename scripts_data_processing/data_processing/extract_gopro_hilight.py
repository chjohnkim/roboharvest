'''
Code derived from: https://github.com/icegoogles/GoPro-Highlight-Parser.git
'''
import os
import click
import struct
import numpy as np

def find_boxes(f, start_offset=0, end_offset=float("inf")):
    """Returns a dictionary of all the data boxes and their absolute starting
    and ending offsets inside the mp4 file.

    Specify a start_offset and end_offset to read sub-boxes.
    """
    s = struct.Struct("> I 4s") 
    boxes = {}
    offset = start_offset
    f.seek(offset, 0)
    while offset < end_offset:
        data = f.read(8)               # read box header
        if data == b"": break          # EOF
        length, text = s.unpack(data)
        f.seek(length - 8, 1)          # skip to next box
        boxes[text] = (offset, offset + length)
        offset += length
    return boxes

def parse_highlights(f, start_offset=0, end_offset=float("inf")):

    inHighlights = False
    inHLMT = False

    listOfHighlights = []

    offset = start_offset
    f.seek(offset, 0)

    while offset < end_offset:
        data = f.read(4)               # read box header
        if data == b"": break          # EOF

        if data == b'High' and inHighlights == False:
            data = f.read(4)
            if data == b'ligh':
                inHighlights = True  # set flag, that highlights were reached

        if data == b'HLMT' and inHighlights == True and inHLMT == False:
            inHLMT = True  # set flag that HLMT was reached

        if data == b'MANL' and inHighlights == True and inHLMT == True:

            currPos = f.tell()  # remember current pointer/position
            f.seek(currPos - 20)  # go back to highlight timestamp

            data = f.read(4)  # readout highlight
            timestamp = int.from_bytes(data, "big")  #convert to integer

            if timestamp != 0:
                listOfHighlights.append(timestamp)  # append to highlightlist

            f.seek(currPos)  # go forward again (to the saved position)

    return np.array(listOfHighlights)/1000  # convert to seconds and return

def examine_mp4(filename):
        
    with open(filename, "rb") as f:
        boxes = find_boxes(f)

        # Sanity check that this really is a movie file.
        def fileerror():  # function to call if file is not a movie file
            print("")
            print("ERROR, file is not a mp4-video-file!")

            os.system("pause")
            exit()

        try:
            if boxes[b"ftyp"][0] != 0:
                fileerror()
        except:
            fileerror()

        moov_boxes = find_boxes(f, boxes[b"moov"][0] + 8, boxes[b"moov"][1])
       
        udta_boxes = find_boxes(f, moov_boxes[b"udta"][0] + 8, moov_boxes[b"udta"][1])

        highlights = parse_highlights(f, udta_boxes[b'GPMF'][0] + 8, udta_boxes[b'GPMF'][1])
        
        return highlights

@click.command()
@click.option('-i', '--input_path', required=True, help='Path to the data directory')
def main(input_path):
    video_path = os.path.join(input_path, 'raw_video.mp4')
    highlights = examine_mp4(video_path)
    save_path = os.path.join(input_path, 'gopro_hilight.csv')
    # Save the highlights to a file by writing hilight in each row
    with open(save_path, 'w') as f:
        for highlight in highlights:
            f.write(f'{highlight}\n')

if __name__=='__main__':
    main()
