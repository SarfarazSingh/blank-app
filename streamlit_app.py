import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import math

mp_pose = mp.solutions.pose

st.title("📏 Tailor Measurement App")

st.write(
    "Upload a front-facing photo and enter your height to estimate body measurements."
)

height_cm = st.number_input("Your height (cm)", min_value=50, max_value=250, value=170)
front_file = st.file_uploader("Front image", type=["jpg", "jpeg", "png"])


def get_landmarks(image: np.ndarray):
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark


def calc_dist(a, b, img_width, img_height):
    ax, ay = int(a.x * img_width), int(a.y * img_height)
    bx, by = int(b.x * img_width), int(b.y * img_height)
    return math.hypot(ax - bx, ay - by)


def compute_measurements(landmarks, img_shape, height_cm):
    img_height, img_width = img_shape[0], img_shape[1]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    # estimate pixel height using average heel position
    heel_y = (left_heel.y + right_heel.y) / 2
    pixel_height = calc_dist(nose, mp_pose.PoseLandmark.LEFT_HEEL, img_width, img_height)
    if pixel_height == 0:
        return None
    scale = height_cm / pixel_height

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

    chest_width = calc_dist(left_shoulder, right_shoulder, img_width, img_height)
    waist_width = calc_dist(left_hip, right_hip, img_width, img_height)
    sleeve_length = calc_dist(left_shoulder, left_wrist, img_width, img_height)
    inseam = calc_dist(left_hip, left_ankle, img_width, img_height)

    # simple circumferences approximations
    chest_circ = chest_width * scale * 2
    waist_circ = waist_width * scale * 2

    return {
        "Chest circumference (cm)": round(chest_circ, 1),
        "Waist circumference (cm)": round(waist_circ, 1),
        "Sleeve length (cm)": round(sleeve_length * scale, 1),
        "Inseam (cm)": round(inseam * scale, 1),
    }

if front_file:
    image = Image.open(front_file)
    img_array = np.array(image)
    landmarks = get_landmarks(img_array)
    if landmarks is None:
        st.error("No person detected in the image.")
    else:
        measurements = compute_measurements(landmarks, img_array.shape, height_cm)
        if measurements:
            st.subheader("Estimated Measurements")
            for k, v in measurements.items():
                st.write(f"{k}: {v} cm")
        else:
            st.error("Could not compute measurements.")

