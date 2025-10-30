import os


# Sprawdzenie, czy istnieje folder data/
if not os.path.exists("data"):
    os.makedirs("data")
    print("Utworzono folder 'data'")
else:
    print("Folder 'data' już istnieje")

# Test biblioteki numpy
try:
    import numpy as np
    print("NumPy działa poprawnie, wersja:", np.__version__)
except ImportError:
    print("Brak biblioteki NumPy — zainstaluj przez 'pip install numpy'")

# Test Kaggle API (czy klucz istnieje i działa)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search="affectnet-yolo-format")
    if datasets:
        print("Kaggle API działa, znaleziono zbiór:", datasets[0].title)
    else:
        print("Kaggle API działa, ale nie znaleziono zbioru.")
except Exception as e:
    print("Nie udało się połączyć z Kaggle API:", e)

print("Test zakończony")
