import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import re
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--output_folder", 
        type=str, 
        help="Output folder"
    )
    
    parser.add_argument(
        "--video_dir", 
        type=str, 
        help="Video directory"
    )
    
    parser.add_argument(
        "--frames", 
        default=1,
        type=int,
        help="Number of frames to be extracted"
    )
    
    args = parser.parse_args()
    
    """train_video_dir = '/aloui/MediaEval/dev-set/sources/'
    test_video_dir = '/aloui/MediaEval/test-set/sources/'

    train_image_dir = 'datasets/ME/images/dc/train/'
    test_image_dir = 'datasets/ME/images/dc/test/'"""

    np.random.seed(42)

    for k, filename in enumerate(tqdm(os.listdir(args.video_dir))):
        if filename.endswith(".webm"):
            vid_id = re.findall(r'\d+', filename)[0]
            vid_id = int(vid_id)
            video_path = os.path.join(args.video_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if args.frames == 1:
                frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.array([0.5]) # middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameIds[0])
                ret, frame = cap.read()
                file_name = os.path.join(args.output_folder, str(vid_id) + '.jpg')
                cv2.imwrite(file_name, frame)
            else:
                frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=args.frames)
                for i, fid in enumerate(frameIds):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ret, frame = cap.read()
                    file_name = os.path.join(args.output_folder, str(vid_id) + '_' + str(i) + '.jpg')
                    cv2.imwrite(file_name, frame)

if __name__ == "__main__":

    main()
