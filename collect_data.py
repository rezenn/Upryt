import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


print("Press 'g' for good posture, 'p' for poor posture, 'q' to quit.")

filename = "posture_data.csv"
file_exists = os.path.isfile(filename)
is_empty = not file_exists or os.path.getsize(
    filename) == 0  # True if new or empty

with open(filename, "a", newline='') as f:
    writer = csv.writer(f)
    if is_empty:
        writer.writerow(["shoulder", "neck", "spine", "symmetry", "label"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            lms = results.pose_landmarks.landmark

            def p(lm): return (int(lm.x * w), int(lm.y * h))

            l_sh, r_sh = p(lms[mp_pose.PoseLandmark.LEFT_SHOULDER]), p(
                lms[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            l_ear, r_ear = p(lms[mp_pose.PoseLandmark.LEFT_EAR]), p(
                lms[mp_pose.PoseLandmark.RIGHT_EAR])
            l_hip, r_hip = p(lms[mp_pose.PoseLandmark.LEFT_HIP]), p(
                lms[mp_pose.PoseLandmark.RIGHT_HIP])

            mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
            mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
            mid_ear = ((l_ear[0] + r_ear[0]) // 2, (l_ear[1] + r_ear[1]) // 2)

            shoulder_angle = calculate_angle(l_sh, r_sh, (r_sh[0], 0))
            neck_angle = calculate_angle(l_ear, l_sh, (l_sh[0], 0))
            spine_angle = calculate_angle(mid_ear, mid_sh, mid_hip)
            symmetry = abs(calculate_angle(l_ear, l_sh, l_hip) -
                           calculate_angle(r_ear, r_sh, r_hip))

            cv2.putText(frame, f"S:{shoulder_angle:.1f} N:{neck_angle:.1f} SP:{spine_angle:.1f} D:{symmetry:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            writer.writerow([shoulder_angle, neck_angle,
                             spine_angle, symmetry, "good"])
            print("Saved GOOD posture")
        elif key == ord('p'):
            writer.writerow([shoulder_angle, neck_angle,
                             spine_angle, symmetry, "poor"])
            print("Saved POOR posture")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
