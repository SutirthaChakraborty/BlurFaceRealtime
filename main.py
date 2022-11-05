import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height,width,channel=image.shape 
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        face_Xmin=int(detection.location_data.relative_bounding_box.xmin*width)
        face_Ymin=int(detection.location_data.relative_bounding_box.ymin*height)
        face_Width=int(detection.location_data.relative_bounding_box.width*width)
        face_height=int(detection.location_data.relative_bounding_box.height*height)
        crop_img = image[face_Ymin:face_Ymin+face_height,face_Xmin:face_Xmin+face_Width]
        blurImg =  cv2.blur(src=crop_img, ksize=(50, 50))
        image[face_Ymin:face_Ymin+face_height,face_Xmin:face_Xmin+face_Width]=blurImg
   
#         mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
