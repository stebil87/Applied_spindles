from pathlib import Path
import pandas as pd
import mne


def read_annot(filename):
    if not Path(f"data/{filename}").exists():
        print(f"{filename} does not exist.")
        return
    return pd.read_csv(f'data/{filename}', sep=r'\s+', 
            header=None, skiprows=1, names=["ts", "duration"])


def load_raw_data():
    raws = {}

    # Patient 1 is not loading
    raws[1] = pd.read_csv(f'data/excerpt1.txt', sep=r'\s+', header=None, skiprows=1, names=["y"]).values

    for i in range(2, 9):
        raws[i] = mne.io.read_raw_edf(f'data/excerpt{i}.edf')
    
    return raws

def load_data():
    signals = {}
    detections_1 = {}
    detections_2 = {}

    raws = load_raw_data()

    channels = {
        2: "CZ-A1",
        3: "C3-A1",
        4: "CZ-A1",
        5: "CZ-A1",
        6: "CZ-A1",
        7: "CZ-A1",
        8: "CZ-A1"
    }

    freqs = [100] + [raws[i].info['sfreq'] for i in range(2, 9)]

    for i in range(1, 9):
        if i == 1:
            signals[i] = pd.DataFrame(raws[i], columns=["y"])
        else:
            signals[i] = pd.DataFrame(raws[i].get_data(channels[i]).ravel(), columns=["y"])
        print(len(signals[i]), freqs[i-1], len(signals[i])/freqs[i-1])
        det_1 = read_annot(f"Visual_scoring1_excerpt{i}.txt")
        det_2 = read_annot(f"Visual_scoring2_excerpt{i}.txt")
        if det_1 is not None:
            detections_1[i] = det_1
        
        if det_2 is not None:
            detections_2[i] = det_2
        
    return signals, detections_1, detections_2


def append_detections(data, det_data, fs):
    data = data.copy()

    # Drop all data which is not recorded by the scorer
    last_record = det_data["ts"].values[-1] + det_data["duration"].values[-1]
    offset = 1
    data = data[data["center_time"] <= last_record + offset]
    
    def is_in_spindle(center):
        return len(det_data[(center > det_data["ts"]) & (center < det_data["ts"] + det_data["duration"])]) > 0
    data["spindle"] = data["center_time"].apply(is_in_spindle).astype(int)
    return data