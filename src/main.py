from ultralytics import YOLO
from numpy import ndarray
import cv2


PERSON_CLASS_ID = 0
HELMET_CLASS_ID = 0
SAFE_COLOR = (40, 220, 80)
DANGER_COLOR = (40, 40, 230)
HELMET_COLOR = (40, 220, 220)


class Detection:

    """A single object detection in xyxy pixel coordinates."""

    def __init__(
        self,
        box: tuple[int, int, int, int],
        confidence: float
    ) -> None:

        self.box = box
        self.confidence = confidence


class SafetyDetector:

    """Detect people and report whether they are wearing safety helmets."""

    def __init__(
        self,
        person_model_path: str = "yolov8n.onnx",
        helmet_model_path: str = "helmet.pt",
    ) -> None:

        self.person_model = YOLO(person_model_path)
        self.helmet_model = YOLO(helmet_model_path)
        self.cap = None
        self.window_name = "Safety Detector"

        self.target_resolution = (1920, 1080)
        self.target_fps = 30
        self.model_input_size = 416
        self.person_confidence_threshold = 0.25
        self.helmet_confidence_threshold = 0.75
        self.min_helmet_inside_person = 0.15
        self.helmet_person_height_ratio = 0.45
        self.window_size = (800, 600)
        self.escape_key = 27

    def initialize_camera(self) -> None:

        """Set up the webcam with the desired resolution and frame rate."""
    
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Capture resolution: {width:.0f}x{height:.0f}")
        print("Detecting people and safety helmets. Press ESC to exit.\n")

    def detections_from_result(
        self,
        result
    ) -> list[Detection]:

        """Convert Ultralytics result boxes to lightweight detections."""

        if result.boxes is None:
            return []
        
        return [
            Detection(
                box=tuple(int(value) for value in box.xyxy[0].tolist()),
                confidence=float(box.conf[0]),
            )
            for box in result.boxes
        ]

    def intersection_area(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int]
    ) -> int:
    
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    
        return max(0, x2 - x1) * max(0, y2 - y1)

    def box_area(
        self,
        box: tuple[int, int, int, int]
    ) -> int:

        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def helmet_belongs_to_person(
        self,
        helmet: Detection,
        person: Detection
    ) -> bool:

        """Return whether a helmet is in the upper portion of a person box."""
        helmet_area = self.box_area(helmet.box)
        if helmet_area == 0:
            return False

        hx1, hy1, hx2, hy2 = helmet.box
        px1, py1, px2, py2 = person.box
        helmet_center_x = (hx1 + hx2) / 2
        helmet_center_y = (hy1 + hy2) / 2
        head_limit = py1 + (py2 - py1) * self.helmet_person_height_ratio

        return (
            self.intersection_area(helmet.box, person.box) / helmet_area
            >= self.min_helmet_inside_person
            and px1 <= helmet_center_x <= px2
            and py1 <= helmet_center_y <= head_limit
        )

    def draw_label(
        self,
        frame: ndarray,
        text: str,
        origin: tuple[int, int],
        color
    ) -> None:
        
        x, y = origin
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y, size[1] + 8)
        cv2.rectangle(frame, (x, label_y - size[1] - 8), (x + size[0] + 8, label_y + baseline), color, -1)
        cv2.putText(frame, text, (x + 4, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 20, 10), 2, cv2.LINE_AA)

    def process_frame(
        self,
        frame: ndarray
    ) -> tuple[ndarray, int, int]:
    
        """Annotate a frame and return people, helmeted people, and helmet status."""
    
        annotated = cv2.resize(frame, (self.model_input_size, self.model_input_size))
        person_result = self.person_model.predict(
            annotated,
            imgsz=self.model_input_size,
            conf=self.person_confidence_threshold,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]
        helmet_result = self.helmet_model.predict(
            annotated,
            imgsz=self.model_input_size,
            conf=self.helmet_confidence_threshold,
            classes=[HELMET_CLASS_ID],
            verbose=False,
        )[0]
        people = self.detections_from_result(person_result)
        helmets = self.detections_from_result(helmet_result)

        helmeted_people = 0
        for person in people:
            has_helmet = any(self.helmet_belongs_to_person(helmet, person) for helmet in helmets)
            color = SAFE_COLOR if has_helmet else DANGER_COLOR
            status = "HELMET ON" if has_helmet else "NO HELMET"
            x1, y1, x2, y2 = person.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            self.draw_label(annotated, f"{status} {person.confidence:.2f}", (x1, y1), color)
            helmeted_people += has_helmet

        for helmet in helmets:
            x1, y1, x2, y2 = helmet.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), HELMET_COLOR, 2)
            self.draw_label(annotated, f"helmet {helmet.confidence:.2f}", (x1, y1), HELMET_COLOR)

        return annotated, len(people), helmeted_people

    def run(self) -> None:

        """Run the real-time helmet compliance detection loop."""
        
        self.initialize_camera()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                # Mirror the webcam before detection so labels stay readable.
                frame = cv2.flip(frame, 1)
                annotated, people, helmeted_people = self.process_frame(frame)
                cv2.putText(annotated, f"People: {people} | Helmet on: {helmeted_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(self.window_name, annotated)
                if cv2.waitKey(1) == self.escape_key:
                    break
        finally:
            self.cleanup()

    def cleanup(self) -> None:

        """Release camera and window resources."""
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released.")


if __name__ == "__main__":

    SafetyDetector().run()
