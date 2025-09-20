import cv2
import threading
import queue
import time
from PIL import Image
from .drawing import draw_keypoints_single_pose
from .posture_tracker import PostureState

class DetectionProcessor:
    def __init__(self, source, pose_estimator, det_conf, skip_frames, posture_tracker=None):
        self.source = source
        self.pose_estimator = pose_estimator
        self.det_conf = det_conf
        self.skip_frames = skip_frames
        self.posture_tracker = posture_tracker

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.running = False

    def _detection_worker(self):
        print("Starting detection worker")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                break

            first_pose = self.pose_estimator.detect(frame, self.det_conf)[0]

            # After getting keypoints for the current frame
            if self.posture_tracker is not None:
                self.posture_tracker.update_live_pose(first_pose['keypoints'])

            if not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(first_pose)

    def start(self):
        print("Starting video processor...")
        self.running = True
        self.worker_thread.start()

        frame_count = 0
        last_result = None

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

            if frame_count % self.skip_frames == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(image.copy())
                curr_posture = self.posture_tracker.classify_posture()
                print("Current Posture is "+str(curr_posture))

            if not self.result_queue.empty():
                last_result = self.result_queue.get()

            display_frame = draw_keypoints_single_pose(last_result, image) if last_result is not None else frame
            cv2.imshow("Posture Panopticon", display_frame)

            # Handle Keystroke
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                self.posture_tracker.record_reference(PostureState.UPRIGHT)
                print("Mapped upright posture reference")
        
        self.stop()

    def stop(self):
        print("Stopping video processor...")
        self.running = False
        if self.worker_thread.is_alive():
            # Empty the queue to allow the worker to exit
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue.put(None)
            self.worker_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Video processor stopped.")
