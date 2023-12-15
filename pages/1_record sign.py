import streamlit as st
import os
import cv2
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


menu = []
dataset_size = 100


def record_data(j):
    cap = cv2.VideoCapture(0)
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" for record the sign', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(
            j), '{}.jpg'.format(counter)), frame)

        counter += 1

    cap.release()
    cv2.destroyAllWindows()


def record_data_now():
    contents = os.listdir(DATA_DIR)
    folders = [item for item in contents if os.path.isdir(
        os.path.join(DATA_DIR, item))]

    # Count the number of folders
    num_folders = len(folders)

    record_data(num_folders)
    print(num_folders)


st.title('mantap')


def create_dataset():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True,
                           min_detection_confidence=0.3)

    DATA_DIR = 'data'

    data = []
    labels = []

    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)

    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

    df = pd.DataFrame(set(labels), columns=['label'])
    f = open('data_df.pickle', 'wb')
    pickle.dump(df, f)
    f.close()


def train_data():
    data_dict = pickle.load(open('data.pickle', 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=0)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()


st.title('Record data page')

record = st.button('record the data')

if record:
    record_data_now()

process = st.button('process the recorded data')
if process:
    create_dataset()
    train_data()
    st.write('data processing is complate')
