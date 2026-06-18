import cv2

class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self._capture = None
        self._metaData = None

        self._capture_video()
        self._load_metadata()

    def __enter__(self):
        """Allows using 'with VideoLoader(path) as loader:'"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guarantees the video releases from RAM, even if the program crashes."""
        self.release()

    def _capture_video(self) -> None:
        self.capture = cv2.VideoCapture(self.path)
        if (not self.capture.isOpened()):
            raise FileNotFoundError(f"Could not open or find the video file at: '{self.path}'")

    def _load_metadata(self) -> None:
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