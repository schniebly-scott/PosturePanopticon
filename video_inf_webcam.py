import argparse
import os
from src.pose_estimator import PoseEstimator
from src.video_processor import DetectionProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default=0,
        help='path to the input video or webcam index'
    )
    parser.add_argument(
        '--det-conf',
        dest='det_conf',
        default=0.3,
        type=float,
        help='detection confidence threshold'
    )
    parser.add_argument(
        '--pose-model',
        dest='pose_model',
        choices=[
            'usyd-community/vitpose-base',
            'usyd-community/vitpose-base-simple',
            'usyd-community/vitpose-base-coco-aic-mpii',
            'usyd-community/vitpose-plus-small',
            'usyd-community/vitpose-plus-base',
            'usyd-community/vitpose-plus-large',
            'usyd-community/vitpose-plus-huge'
        ],
        default='usyd-community/vitpose-base'
    )
    parser.add_argument(
        '--skip-frames',
        dest='skip_frames',
        default=100,
        type=int,
        help='Number of frames to skip between detections'
    )
    args = parser.parse_args()

    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    pose_estimator = PoseEstimator(pose_model_name=args.pose_model)
    detection_processor = DetectionProcessor(
        source=source,
        pose_estimator=pose_estimator,
        det_conf=args.det_conf,
        skip_frames=args.skip_frames
    )
    detection_processor.start()

if __name__ == '__main__':
    main()