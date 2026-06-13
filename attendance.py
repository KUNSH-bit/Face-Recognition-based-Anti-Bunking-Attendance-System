import cv2
import face_recognition
import pandas as pd
import os
from datetime import datetime

ESP32_URL = "http://192.168.1.100:81/stream"

known_faces = []
known_names = []

for file in os.listdir("known_faces"):
    if file.endswith(".jpg") or file.endswith(".png"):

        image = face_recognition.load_image_file(
            f"known_faces/{file}"
        )

        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_faces.append(encoding[0])
            known_names.append(file.split('.')[0])

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(
        columns=["Name", "Entry", "Exit"]
    ).to_csv(attendance_file, index=False)

cap = cv2.VideoCapture(ESP32_URL)

person_positions = {}

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(
        rgb,
        face_locations
    )

    for encoding, location in zip(
        face_encodings,
        face_locations
    ):

        matches = face_recognition.compare_faces(
            known_faces,
            encoding
        )

        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        top, right, bottom, left = location

        center_x = (left + right) // 2

        if name != "Unknown":

            if name not in person_positions:
                person_positions[name] = center_x

            else:

                start_x = person_positions[name]

                direction = None

                if center_x - start_x > 150:
                    direction = "ENTRY"

                elif start_x - center_x > 150:
                    direction = "EXIT"

                if direction:

                    df = pd.read_csv(attendance_file)

                    now = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    if direction == "ENTRY":

                        if not (
                            (df["Name"] == name)
                        ).any():

                            new_row = {
                                "Name": name,
                                "Entry": now,
                                "Exit": ""
                            }

                            df.loc[len(df)] = new_row

                    else:

                        rows = df[df["Name"] == name]

                        if len(rows) > 0:

                            idx = rows.index[-1]

                            if pd.isna(
                                df.loc[idx, "Exit"]
                            ) or df.loc[idx, "Exit"] == "":

                                df.loc[idx, "Exit"] = now

                    df.to_csv(
                        attendance_file,
                        index=False
                    )

                    del person_positions[name]

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
