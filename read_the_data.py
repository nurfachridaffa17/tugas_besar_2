import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

file_path = 'data_cuaca_edit.csv'

data = pd.read_csv(file_path)

# Menghapus kolom yang tidak diperlukan
data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Summary']]

# Encode label
label_encoder = LabelEncoder()
data['Summary'] = label_encoder.fit_transform(data['Summary'])

# Memisahkan fitur dan label
X = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
y = data['Summary']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Melatih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model KNN dan Naive Bayes telah disimpan.")

