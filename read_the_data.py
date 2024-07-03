import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# File path to the CSV file
file_path = 'data_cuaca_edit.csv'

# Load data
data = pd.read_csv(file_path)

# Menghapus kolom yang tidak diperlukan
data = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Summary', 'Daily Summary']]

# Encode Summary dan Daily Summary
label_encoder_summary = LabelEncoder()
label_encoder_daily_summary = LabelEncoder()

data['Summary'] = label_encoder_summary.fit_transform(data['Summary'])
data['Daily Summary'] = label_encoder_daily_summary.fit_transform(data['Daily Summary'])

# Memisahkan fitur dan label
X = data[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
y = data['Daily Summary']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Melatih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Evaluasi model KNN
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy KNN: {accuracy_knn:.2f}')

# Evaluasi model Naive Bayes
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Accuracy Naive Bayes: {accuracy_nb:.2f}')

# Simpan model ke file
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(label_encoder_summary, 'label_encoder_summary.pkl')
joblib.dump(label_encoder_daily_summary, 'label_encoder_daily_summary.pkl')

print("Model KNN dan Naive Bayes telah disimpan.")