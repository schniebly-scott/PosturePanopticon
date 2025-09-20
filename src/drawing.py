import cv2
import numpy as np
import matplotlib

EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
]

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
        for ie, e in enumerate(EDGES):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                ie/float(len(EDGES)), 1.0, 1.0
            ])
            rgb = rgb*255
            # join the keypoint pairs to draw the skeletal structure
            cv2.line(image, tuple(map(int, keypoints[e[0]])), 
                     tuple(map(int, keypoints[e[1]])), 
                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image

def draw_keypoints_single_pose(output, image):
    """
    :param output: single Output from the keypoint detector.
    :param image: Image in PIL Image format.

    Returns:
        image: Annotated image Numpy array format.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    keypoints = output['keypoints'].cpu().detach().numpy()
    # proceed to draw the lines if the confidence score is above 0.9
    keypoints = keypoints[:, :].reshape(-1, 2)
    for p in range(keypoints.shape[0]):
        # draw the keypoints
        cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                    3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        # uncomment the following lines if you want to put keypoint number
        # cv2.putText(image, f'{p}', (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for ie, e in enumerate(EDGES):
        # get different colors for the edges
        rgb = matplotlib.colors.hsv_to_rgb([
            ie/float(len(EDGES)), 1.0, 1.0
        ])
        rgb = rgb*255
        # join the keypoint pairs to draw the skeletal structure
        cv2.line(image, tuple(map(int, keypoints[e[0]])), 
                 tuple(map(int, keypoints[e[1]])), 
                 tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image
