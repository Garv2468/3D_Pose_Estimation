import cv2

class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self.capture = None
        self.metaData = None

        self.capture_video()
        self.load_metadata()

    def capture_video(self) -> None:
        self.capture = cv2.VideoCapture(self.path)
        if (not self.capture.isOpened()):
            raise FileNotFoundError(f"Could not open or find the video file at: '{self.path}'")

    def load_metadata(self) -> None:
        assert self.capture is not None, "Capture object is missing."

        self.metaData = {
            "fps" : int(self.capture.get(cv2.CAP_PROP_FPS)),
            "width" : int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height" : int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames" : int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def get_metadata(self) -> dict:
        if (self.metaData is None):
            raise ValueError("Metadata not found. Run load_metadata() first.")
        return self.metaData

    def get_video_capture(self) -> cv2.VideoCapture:
        if (self.capture is None):
            raise ValueError("Video Capture not Found. Run capture_video() first.")
        return self.capture

    def read_frame(self) -> tuple:
        assert self.capture is not None, "Capture object is missing."
            
        return self.capture.read()    

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None