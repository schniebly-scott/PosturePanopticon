import torch
import time
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class PoseEstimator:
    def __init__(self, pose_model_name='usyd-community/vitpose-base', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load detector
        self.person_image_processor = AutoProcessor.from_pretrained('PekingU/rtdetr_r50vd_coco_o365')
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            'PekingU/rtdetr_r50vd_coco_o365', device_map=self.device
        )

        # Load ViTPose
        print(f"Loading Pose Model: {pose_model_name}")
        self.pose_model_name = pose_model_name
        self.pose_image_processor = AutoProcessor.from_pretrained(pose_model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_name, device_map=self.device)

    def detect(self, image, det_conf=0.3):
        person_boxes = self._detect_objects(image, det_conf)
        if len(person_boxes) == 0:
            return None
        poses = self._detect_pose(image, person_boxes)
        return poses

    def _detect_objects(self, image, det_conf):
        """
        :param image: Image in PIL image format.
        Returns:
            person_boxes: Bboxes of persons in [x, y, w, h] format.
        """
        inputs = self.person_image_processor(images=image, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.person_model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)])
        results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=det_conf
        )

        result = results[0]
        person_boxes_xyxy = result['boxes'][result['labels'] == 0]
        person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()

        person_boxes = person_boxes_xyxy.copy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

        return person_boxes

    def _detect_pose(self, image, person_boxes):
        """
        :param image: Image in PIL image format.
        :param person_boxes: Batched person boxes in [[x, y, w, h], ...] format.
        """
        inputs = self.pose_image_processor(image, boxes=[person_boxes], return_tensors='pt').to(self.device)
        dataset_index = torch.tensor([0], device=self.device)

        if 'plus' in self.pose_model_name:
            with torch.no_grad():
                outputs = self.pose_model(**inputs, dataset_index=dataset_index)
        else:
            with torch.no_grad():
                outputs = self.pose_model(**inputs)

        pose_results = self.pose_image_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )

        if not pose_results:
            return None

        return pose_results[0]
