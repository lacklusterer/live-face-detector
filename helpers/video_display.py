import cv2
import matplotlib.pyplot as plt

def display_frame(frame, window_name="Processed Video Stream"):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(window_name)
    plt.axis("off")
    plt.pause(0.001)
