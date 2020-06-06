import cv2
import numpy as np
import dlib
from math import hypot
# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
nose_image = cv2.imread("pig_nose.png")
wrestling_image = cv2.imread("wrestling_mask.png")
clown_image = cv2.imread("clown_mask.png")
dog_image = cv2.imread("dog_mask.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)
# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("MASK PROJECT")
print("For Pig Nose \t\t press \t '1'")
print("For Wrestling Mask \t press \t '2'")
print("For Clown Mask \t\t press \t '3'")
print("For Dog Mask \t\t press \t '4'")

def rotate_image(image, angle):
    angle =( angle*180)/3.14
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def pig_mask(landmarks, nose_image, nose_mask):
    top_nose = (landmarks.part(29).x, landmarks.part(29).y)
    center_nose = (landmarks.part(30).x, landmarks.part(30).y)
    left_nose = (landmarks.part(31).x, landmarks.part(31).y)
    right_nose = (landmarks.part(35).x, landmarks.part(35).y)
    height_angle = left_nose[1]-right_nose[1]
    width_angle = right_nose[0]-left_nose[0]
    angle = np.arctan(height_angle/width_angle)
    nose_width = int(hypot(left_nose[0] - right_nose[0],
                       left_nose[1] - right_nose[1]) * 2.5)
    nose_height = int(nose_width * 0.77)
    # New nose position
    top_left = (int(center_nose[0] - nose_width / 2),
                          int(center_nose[1] - nose_height / 2))
    bottom_right = (int(center_nose[0] + nose_width / 2),
                   int(center_nose[1] + nose_height / 2))
            # Adding the new nose
    nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
    nose_pig = rotate_image(nose_pig, angle)
    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
    _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
    nose_area = frame[top_left[1]: top_left[1] + nose_height,
                top_left[0]: top_left[0] + nose_width]
    nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
    final_nose = cv2.add(nose_area_no_nose, nose_pig)
    frame[top_left[1]: top_left[1] + nose_height,
                top_left[0]: top_left[0] + nose_width] = final_nose
    cv2.imshow("Nose area", nose_area)
    cv2.imshow("Nose pig", nose_pig)
    cv2.imshow("Nose mask", nose_mask)
    cv2.imshow("final nose", final_nose)

def full_mask(landmarks, wrestling_image, nose_mask, width_multiply, height_multiply, y_transform=0):
    forhead = (int((landmarks.part(21).x+landmarks.part(22).x)/2), int((landmarks.part(21).y+landmarks.part(22).y)/2))
    chin = (landmarks.part(8).x, landmarks.part(8).y)
    # cv2.circle(frame, (chin[0], chin[1]), 4, (0, 0, 255), -1)
    left = (landmarks.part(0).x, landmarks.part(0).y)
    right = (landmarks.part(16).x, landmarks.part(16).y)
    height_angle = left[1]-right[1]
    width_angle = right[0]-left[0]
    angle = np.arctan(height_angle/width_angle)
    width = int(hypot(left[0] - right[0],
                       left[1] - right[1])*width_multiply)
    height = int(hypot(forhead[0] - chin[0],
                       forhead[1] - chin[1])*height_multiply)
    # New nose position
    top_left = (int(forhead[0] - width / 2),
                          int(chin[1]-height+y_transform))
    bottom_right = (int(forhead[0] + width / 2),
                   int(chin[1])+y_transform)
            # Adding the new nose
    mask_wrest = cv2.resize(wrestling_image, (width, height))
    mask_wrest = rotate_image(mask_wrest, angle)
    mask_wrest_gray = cv2.cvtColor(mask_wrest, cv2.COLOR_BGR2GRAY)
    _, mask_mask = cv2.threshold(mask_wrest_gray, 25, 255, cv2.THRESH_BINARY_INV)
    mask_area = frame[top_left[1]: top_left[1] + height,
                top_left[0]: top_left[0] + width]
    mask_area_no_mask = cv2.bitwise_and(mask_area, mask_area, mask=mask_mask)
    final_mask = cv2.add(mask_area_no_mask, mask_wrest)
    frame[top_left[1]: top_left[1] + height,
                top_left[0]: top_left[0] + width] = final_mask
    # cv2.imshow("Nose area", nose_area)
    # cv2.imshow("Nose pig", nose_pig)
    # cv2.imshow("Nose mask", nose_mask)
    cv2.imshow("final nose", final_mask)
    
mask={
    'pig': 1,
    'wrestling': 2,
    'clown': 3,
    'dog': 4,
}
mask_number = 1
while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        if mask_number == 1:
            pig_mask(landmarks, nose_image, nose_mask)
        elif mask_number == 2:
            full_mask(landmarks, wrestling_image, nose_mask,1.2,1.7)
        elif mask_number == 3:
            full_mask(landmarks, clown_image, nose_mask,1.8, 2.1,20)
        elif mask_number == 4:
            full_mask(landmarks, dog_image, nose_mask,2.1, 2.1,20)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 49:
        mask_number = mask['pig']
        nose_mask = np.zeros((rows, cols), np.uint8)
        print("pig", mask_number)
    elif key == 50:
        mask_number = mask['wrestling']
        nose_mask = np.zeros((rows, cols), np.uint8)
        print("wrestling", mask_number)
    elif key == 51:
        mask_number = mask['clown']
        nose_mask = np.zeros((rows, cols), np.uint8)
        print("clown", mask_number)
    elif key == 52:
        mask_number = mask['dog']
        nose_mask = np.zeros((rows, cols), np.uint8)
        print("dog", mask_number)