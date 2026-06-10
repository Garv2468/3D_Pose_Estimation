import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Couldn't open")

    metadata = {
        "fps" : cap.get(cv2.CAP_PROP_FPS),
        "width" : cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height" : cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "total_frames" : cap.get(cv2.CAP_PROP_FRAME_COUNT)
    }

    return cap, metadata
    
def frame_generator(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret :
            break
        yield frame

def release_video(cap):
    cap.release()