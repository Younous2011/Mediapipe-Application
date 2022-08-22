import cv2
import mediapipe as mp
import numpy as np
import skimage
from skimage.draw import polygon
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

MOUTH_INDICES_INTERIOR = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
font = cv2.FONT_HERSHEY_SIMPLEX

# My functions
def to_numpy(data):
    data_l = []
    for d in data:
        el = [d["x"], d["y"]]
        data_l.append(el)

    data_np = np.array(data_l)
    return data_np

def parse_landmarks(face_landmark):
  landmark_list = []
  for lm in list(face_landmark.landmark):
    lm_dict = {
      "x": lm.x,
      "y": lm.y,
      "z": lm.z,
      "presence": lm.presence,
      "visibility": lm.visibility
    }
    landmark_list.append(lm_dict)

  return landmark_list


def extract_mouth_coordinates(shape, data, couple):
    x = data[couple, 0] * shape[1]
    y = data[couple, 1] * shape[0]
    return x, y

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    """
      À l'intèreur de la boucle
      image[rr, cc] = [255, 0, 0] #red
    """
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        face_landlist = parse_landmarks(face_landmarks)
        data_np = to_numpy(face_landlist)
        x, y = extract_mouth_coordinates(image.shape, data_np, MOUTH_INDICES_INTERIOR)
        rr, cc = polygon(y, x)
        area_poly = PolyArea(rr, cc)
        open_closed = "fermee"
        if area_poly > 20:
          open_closed = "ouverte"
        else:
          open_closed = "fermee"
        if (rr < 480).all() and (cc < 640).all():
          image[rr, cc] = [255, 0, 0]
        # print("face_landlist : {}".format(face_landlist))
        # print("data_np : {}".format(data_np))
        # print("x : {}, y : {}".format(x, y))
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION, # -> Face
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
            # image=image,
            # landmark_list=face_landmarks,
            # connections=mp_face_mesh.FACEMESH_CONTOURS, # -> Contours
            # landmark_drawing_spec=None,
            # connection_drawing_spec=mp_drawing_styles
            # .get_default_face_mesh_contours_style())
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.putText(image, str(area_poly) + "px2" + " bouche " + open_closed,(0,25), font, 1,(255,255,255),2)
    cv2.imshow('MediaPipe Face Mesh', image)#cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
