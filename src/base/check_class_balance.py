import os
from collections import Counter
import matplotlib.pyplot as plt

TRAIN_LABEL_DIR = "data/affectnet_raw/YOLO_format/train/labels"

RESULTS_DIR = "experiments/analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)
TXT_PATH = os.path.join(RESULTS_DIR, "class_counts_train.txt")
PNG_PATH = os.path.join(RESULTS_DIR, "class_counts_train.png")

def load_class_counts(label_dir):
    counter = Counter()
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counter[cls_id] += 1
    return counter

def main():
    if not os.path.isdir(TRAIN_LABEL_DIR):
        print(f"[ERROR] Nie znaleziono folderu z labelami: {TRAIN_LABEL_DIR}")
        return

    counts = load_class_counts(TRAIN_LABEL_DIR)

    full_counts = {i: counts.get(i, 0) for i in range(8)}

    print("[INFO] Liczba przykładów na klasę (train):")
    for cls_id, cnt in full_counts.items():
        print(f"klasa {cls_id}: {cnt}")

    # zapis do txt
    with open(TXT_PATH, "w") as f:
        f.write("Class counts for train set\n")
        for cls_id, cnt in full_counts.items():
            f.write(f"class_{cls_id}={cnt}\n")

    classes = list(full_counts.keys())
    values = list(full_counts.values())

    plt.figure(figsize=(6, 4))
    plt.bar(classes, values)
    plt.xlabel("klasa")
    plt.ylabel("liczba przykładów")
    plt.title("Rozkład klas w zbiorze treningowym")
    plt.xticks(classes)
    plt.tight_layout()
    plt.savefig(PNG_PATH)
    plt.close()

    print(f"[OK] Zapisano wyniki do {TXT_PATH}")
    print(f"[OK] Zapisano wykres do {PNG_PATH}")

if __name__ == "__main__":
    main()
