import cv2

from ultralytics import YOLO

model = YOLO("best_V4_task47_yolo8m___AP2_4#2.pt")

#source = 'https://youtu.be/kjem5-kanuc'
source = 1 # внутренний номер видеокамеры в системе

cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

# Loop through the video frames
while True:

    # Read a frame from the video
    success, frame = cap.read()
 #   print(success)
 #   print(type(frame))
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        # Visualize the results on the frame
        frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("Видео", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
