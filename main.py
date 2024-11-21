import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fonction pour calculer la distance euclidienne
def distance_euclidienne(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Fonction pour calculer la distance de Manhattan
def distance_manhattan(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

def k_nearest_neighbors(X_train, y_train, X_test, k):
    # X_train : données d'entraînement
    # y_train : étiquettes des données d'entraînement
    # X_test : données de test
    # k : nombre de voisins à considérer

    y_pred = []
    
    for test_point in X_test:
        distances = []
        
        # Calcul des distances entre le point de test et tous les points d'entraînement
        for i in range(len(X_train)):
            dist = distance_manhattan(test_point, X_train[i])
            distances.append((dist, y_train[i]))  # Stocker la distance et la classe associée
        
        # Trier les distances et sélectionner les k plus proches voisins
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        
        # Extraire les classes des k plus proches voisins
        classes = [neighbor[1] for neighbor in top_k]
        
        # Déterminer la classe la plus fréquente parmi les k voisins
        classe_predite = Counter(classes).most_common(1)[0][0]
        y_pred.append(classe_predite)
    
    return np.array(y_pred)

def load_dataset(filename):
    X, y = [], []
    with open(filename, 'r') as file:
        for line in file:
            try:
                values = line.strip().split(',')
                X.append([float(value) for value in values[1:-1]]) #On retire les id et la classe
                y.append(int(values[-1]))
            except:
                continue
    return np.array(X), np.array(y)

def load_test_set(filename):
    X = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                values = line.strip().split(',')
                X.append([float(value) for value in values[1:]]) #On retire les id
            except:
                continue
    return np.array(X)

X_train, y_train = load_dataset("train.csv")
X_test = load_test_set("test.csv")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 3
predictions = k_nearest_neighbors(X_train, y_train, X_test, k)
print("Prédictions pour le fichier de test:", predictions)


def write_predictions_to_csv(input_csv, output_csv, predictions):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        if len(header) < 2:
            header.append("Prediction")
        writer.writerow(header)
    
        for i, row in enumerate(reader):
            if i < len(predictions): 
                if len(row) < 2:
                    row.append(predictions[i])
                else:
                    row[1] = predictions[i]
            writer.writerow(row)

input_csv = "sample_submission.csv"
output_csv = "sample_submission_with_predictions4.csv"
write_predictions_to_csv(input_csv, output_csv, predictions)

def calculate_difference_percentage(file1, file2):
    """
    Compare les colonnes 'Prediction' de deux fichiers CSV de soumission
    et calcule le pourcentage de différence.
    
    Args:
        file1 (str): Chemin du premier fichier CSV.
        file2 (str): Chemin du deuxième fichier CSV.
    
    Returns:
        float: Pourcentage de différence entre les deux fichiers.
    """
    predictions1 = {}
    predictions2 = {}

    # Lecture du premier fichier
    with open(file1, 'r') as f1:
        reader1 = csv.reader(f1)
        next(reader1)  # Ignorer l'en-tête
        for row in reader1:
            predictions1[row[0]] = row[1]  # Stocker {id: prediction}

    # Lecture du deuxième fichier
    with open(file2, 'r') as f2:
        reader2 = csv.reader(f2)
        next(reader2)  # Ignorer l'en-tête
        for row in reader2:
            predictions2[row[0]] = row[1]  # Stocker {id: prediction}

    # Comparer les prédictions
    total = len(predictions1)
    differences = 0

    for id, pred1 in predictions1.items():
        pred2 = predictions2.get(id, None)
        if pred2 is None or pred1 != pred2:  # Si le prédiction diffère ou est manquante
            differences += 1

    # Calcul du pourcentage de différence
    difference_percentage = (differences / total) * 100
    return difference_percentage

# Exemple d'utilisation
file1 = "sample_submission_with_predictions3.csv"
file2 = "sample_submission_with_predictions4.csv"

diff_percentage = calculate_difference_percentage(file1, file2)
print(f"Pourcentage de différence entre les soumissions : {diff_percentage:.2f}%")