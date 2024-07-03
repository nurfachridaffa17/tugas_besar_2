import tkinter as tk
from tkinter import ttk
from datetime import datetime
import joblib

knn_model = joblib.load('knn_model.pkl')
nb_model = joblib.load('nb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class WeatherPrediction(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.selected_algorithm = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        # Label dan ComboBox untuk pemilihan algoritma
        self.label_algo = tk.Label(self, text="Pilih Algoritma:")
        self.label_algo.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.combo_algo = ttk.Combobox(self, textvariable=self.selected_algorithm)
        self.combo_algo['values'] = ["Naive Bayes", "K-Nearest Neighbors (KNN)"]
        self.combo_algo.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        # Entry fields untuk Temperature
        self.label_temp = tk.Label(self, text="Temperature (°C):")
        self.label_temp.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        self.entry_temp = tk.Entry(self)
        self.entry_temp.grid(row=1, column=1, padx=10, pady=5, sticky='w')

        # Entry fields untuk Humidity
        self.label_humidity = tk.Label(self, text="Humidity (%):")
        self.label_humidity.grid(row=2, column=0, padx=10, pady=5, sticky='w')

        self.entry_humidity = tk.Entry(self)
        self.entry_humidity.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        # Entry fields untuk Wind Speed
        self.label_wind = tk.Label(self, text="Wind Speed (km/h):")
        self.label_wind.grid(row=3, column=0, padx=10, pady=5, sticky='w')

        self.entry_wind = tk.Entry(self)
        self.entry_wind.grid(row=3, column=1, padx=10, pady=5, sticky='w')

        # Entry fields untuk Pressure
        self.label_pressure = tk.Label(self, text="Pressure (hPa):")
        self.label_pressure.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        self.entry_pressure = tk.Entry(self)
        self.entry_pressure.grid(row=4, column=1, padx=10, pady=5, sticky='w')

        # Tombol untuk menampilkan prediksi
        self.predict_button = tk.Button(self, text="Prediksi", command=self.predict_weather)
        self.predict_button.grid(row=5, columnspan=2, pady=10)

        # Label untuk menampilkan hasil prediksi
        self.result_label = tk.Label(self, text="")
        self.result_label.grid(row=6, columnspan=2, padx=10, pady=10)

    def predict_weather(self):
        try:
            temperature = float(self.entry_temp.get())
            humidity = float(self.entry_humidity.get())
            wind_speed = float(self.entry_wind.get())
            pressure = float(self.entry_pressure.get())
            selected_algo = self.selected_algorithm.get()

            features = [[temperature, humidity, wind_speed, pressure]]

            if selected_algo == "K-Nearest Neighbors (KNN)":
                prediction = knn_model.predict(features)[0]
            elif selected_algo == "Naive Bayes":
                prediction = nb_model.predict(features)[0]
            else:
                self.result_label.config(text="Pilih algoritma terlebih dahulu.")
                return

            predicted_label = label_encoder.inverse_transform([prediction])[0]

            self.result_label.config(text=f"Prediksi Cuaca: {predicted_label}")

        except ValueError:
            self.result_label.config(text="Masukkan semua nilai dengan benar!")

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Prediksi Cuaca")
        self.geometry("400x300")

        self.weather_prediction_frame = WeatherPrediction(self)
        self.weather_prediction_frame.pack(padx=10, pady=10)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
