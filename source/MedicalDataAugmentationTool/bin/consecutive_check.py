import numpy as np
import pandas as pd
from pathlib import Path

def consecutive_checker(output_folder):
    output_folder = Path(output_folder)
    landmarks = pd.read_csv(output_folder / "landmarks.csv", header=None).to_numpy()[0]
    valid_landmarks = pd.read_csv(output_folder / "valid_landmarks.csv", header=None).to_numpy()[0]
    # file = landmarks[0]
    coords = landmarks[1:]
    valid_landmarks = pd.DataFrame({'label': valid_landmarks[1:]})
    df_landmarks = []
    for j in range(0, len(coords), 3):
        df_landmarks.append(coords[j:j+3])

    df_landmarks = pd.DataFrame(df_landmarks, columns=['x', 'y', 'z'])
    valid_landmarks = valid_landmarks.merge(df_landmarks, how='left', left_on='label', right_index=True)
    valid_landmarks_sort = valid_landmarks.sort_values('z', ascending=False)
    if not valid_landmarks.equals(valid_landmarks_sort) or not np.array_equal(valid_landmarks_sort['label'], list(range(valid_landmarks['label'].iloc[0], valid_landmarks['label'].iloc[-1]+1))):
        print('\nSorting vertebrae\n')
        new_landmarks = np.full(len(landmarks), 'nan', dtype=object)
        new_valid_landmarks = np.zeros(valid_landmarks_sort.shape[0] + 1, dtype=object)
        new_landmarks[0] = landmarks[0]
        new_valid_landmarks[0] = landmarks[0]
        new_valid_landmarks[1:] = range(0, valid_landmarks_sort.shape[0])
        j = 1
        for i in range(0, valid_landmarks_sort.shape[0]):
            new_landmarks[j : j+3] = valid_landmarks_sort.iloc[i, 1:]
            j += 3
            ...
        pd.DataFrame(new_landmarks).T.to_csv(output_folder / 'landmarks.csv', index=False, header=False)
        pd.DataFrame(new_valid_landmarks).T.to_csv(output_folder / 'valid_landmarks.csv', index=False, header=False)