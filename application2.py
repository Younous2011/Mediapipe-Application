import cv2
import mediapipe as mp
import logging
import json
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# My functions
def write_json(dictt, path:str, indent=4):
    json_object = json.dumps(dictt, indent=indent)
    with open(path, "w") as outfile:
        outfile.write(json_object)

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

def output_path_fn(path, ext):
    basename = path.split("/")[-1]
    basename_n = basename.split(".")[0]
    basename_json = basename_n + "." + ext
    path_json = "./output/" + basename_json
    return path_json

# For static images:

IMAGE_FILES = ['./imgs/younous.png', './imgs/opened_mouth.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=300)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    logging.info(str(results))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print("#")
      parsed_landmarks_list = parse_landmarks(face_landmarks)
      print(parsed_landmarks_list[0])
      path_json = output_path_fn(file, "json")
      write_json(parsed_landmarks_list, path_json)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    path_png = output_path_fn(file, "png")
    cv2.imwrite(path_png, annotated_image)