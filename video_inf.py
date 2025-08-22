import torch
import numpy as np
import cv2
import matplotlib
import argparse
import os
import time

from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    default=0,
    help='path to the input video'
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
args = parser.parse_args()

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_fps = int(cap.get(5)) 

save_name = args.input.split(os.path.sep)[-1].split('.')[0]
# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f'{OUT_DIR}/{save_name}.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    video_fps, 
    (frame_width, frame_height)
)

# Load detector.
person_image_processor = AutoProcessor.from_pretrained(
    'PekingU/rtdetr_r50vd_coco_o365'
)
person_model = RTDetrForObjectDetection.from_pretrained(
    'PekingU/rtdetr_r50vd_coco_o365', device_map=device
)

# Load ViTPose.
print(f"Pose Model: {args.pose_model}")
image_processor = AutoProcessor.from_pretrained(args.pose_model)
model = VitPoseForPoseEstimation.from_pretrained(args.pose_model, device_map=device)

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
]

def detect_objects(image):
    """
    :param image: Image in PIL image format.

    Returns:
        person_boxes: Bboxes of persons in [x, y, w, h] format.
    """
    det_time_start = time.time()

    inputs = person_image_processor(
        images=image, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    
    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=args.det_conf
    )

    det_time_end = time.time()

    det_fps = 1 / (det_time_end-det_time_start)
    
    # Extract the first result, as we can pass multiple images at a time.
    result = results[0]
    
    # In COCO dataset, humans labels have index 0.
    person_boxes_xyxy = result['boxes'][result['labels'] == 0]
    person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()
    
    # Convert boxes from (x1, y1, x2, y2) to (x1, y1, w, h) format.
    person_boxes = person_boxes_xyxy.copy()
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    return person_boxes, det_fps

def detect_pose(image, person_boxes):
    """
    :param image: Image in PIL image format.
    :param person_bboxes: Batched person boxes in [[x, y, w, h], ...] format.
    """
    pose_time_start = time.time()

    inputs = image_processor(
        image, boxes=[person_boxes], return_tensors='pt'
    ).to(device)
    
    dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)

    if len(person_boxes) != 0:
        if 'plus' in args.pose_model:
            with torch.no_grad():
                outputs = model(**inputs, dataset_index=dataset_index)
        else:
            with torch.no_grad():
                outputs = model(**inputs)
        
        pose_results = image_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )

    pose_time_end = time.time()

    pose_fps = 1 / (pose_time_end-pose_time_start)

    if len(person_boxes) == 0:
        return [], pose_fps

    image_pose_result = pose_results[0]
    
    return image_pose_result, pose_fps

def draw_keypoints(outputs, image):
    """
    :param outputs: Outputs from the keypoint detector.
    :param image: Image in PIL Image format.

    Returns:
        image: Annotated image Numpy array format.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # the `outputs` is list which in-turn contains the dictionaries 
    for i, pose_result in enumerate(outputs):
        keypoints = pose_result['keypoints'].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        keypoints = keypoints[:, :].reshape(-1, 2)
        for p in range(keypoints.shape[0]):
            # draw the keypoints
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                        3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # uncomment the following lines if you want to put keypoint number
            # cv2.putText(image, f'{p}', (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for ie, e in enumerate(edges):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                ie/float(len(edges)), 1.0, 1.0
            ])
            rgb = rgb*255
            # join the keypoint pairs to draw the skeletal structure
            cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                    (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                    tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)

        start_time = time.time()

        bboxes, det_fps = detect_objects(image=image)
        image_pose_result, pose_fps = detect_pose(image=image, person_boxes=bboxes)

        result = draw_keypoints(image_pose_result, image)

        end_time = time.time()

        forward_pass_time = end_time - start_time
            
        # Get the current fps.
        fps = 1 / (forward_pass_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

        cv2.putText(
            result,
            f"FPS: {fps:0.1f} | Pose FPS: {pose_fps:0.1f} | Detection FPS: {det_fps:0.1f}",
            (15, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        out.write(result)

        cv2.imshow('Prediction', result)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

 # Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

avg_fps = total_fps/frame_count
print(f"Average FPS: {avg_fps}")