import cv2
import logging
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

@dataclass
class VideoConfig:
    """Configuration settings for video capture and display."""
    width: int = 640
    height: int = 480
    fps: float = 30.0
    window_name: str = "Video Stream"

class VideoProcessor:
    """Handles video capture, processing, and display operations."""
    
    def __init__(self, config: VideoConfig, debug: bool = False):
        self.config = config
        self._setup_logging(debug)
    
    def _setup_logging(self, debug: bool) -> None:
        """Configure logging based on debug mode."""
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def _video_capture(self):
        """Context manager for video capture device."""
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                raise RuntimeError("Failed to open webcam")
            
            # Configure capture properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Verify the actual capture FPS
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            self.logger.debug(f"Capture FPS set to: {actual_fps}")
            
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Process a single frame."""
        from models.yunet.yunet import process
        return process(frame)

    def run(self) -> None:
        """Main processing loop."""
        try:
            with self._video_capture() as cap:
                self._process_and_display(cap)
        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
            raise

    def _process_and_display(self, cap) -> None:
        """Process frames from capture and display them."""
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                break
            
            processed_frame = self.process_frame(frame)
            
            # Display the processed frame
            cv2.imshow(self.config.window_name, processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User requested quit")
                break

