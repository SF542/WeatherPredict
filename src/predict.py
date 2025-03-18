import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(12)
        ])
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        return model

    def train(self, X_train, y_train, epochs=300, batch_size=4):
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict_next_year(self, last_year_data):
        return self.model.predict(np.array([last_year_data]))[0]

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip().split(',')
    data = [float(x) for x in data]
    return data

def prepare_dataset(data):
    data = np.array(data)
    num_years = len(data) // 12
    data = data[:num_years*12].reshape(-1, 12)
    X = data[:-1]
    y = data[1:]
    return X, y

def plot_predictions(predictions):
    months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
    plt.figure(figsize=(14, 7))
    plt.plot(months, predictions, marker='o', linestyle='-', color='b')
    plt.title('Прогноз средней температуры на следующий год', fontsize=14)
    plt.xlabel('Месяц', fontsize=12)
    plt.ylabel('Температура, °C', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast.png')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Обучение', linewidth=2)
    plt.plot(history.history['val_loss'], label='Валидация', linewidth=2)
    plt.title('График потерь во время обучения', fontsize=14)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    file_path = input("Введите путь к входному датасету: ").strip()
    raw_data = load_data(file_path)
    X, y = prepare_dataset(raw_data)
    choice = input("Хотите создать новую модель (1) или использовать существующую (2)? Введите 1 или 2: ").strip()
    predictor = WeatherPredictor()
    
    if choice == '1':
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )
        predictor.model = predictor.build_model(X_train.shape[1])
        predictor.scaler_X = scaler_X
        predictor.scaler_y = scaler_y
        predictor.train(X_train, y_train)
        loss, mae = predictor.evaluate(X_test, y_test)
        print(f"Test MAE: {mae:.2f}°C")
        save_dir = input("Введите папку для сохранения модели и scaler'ов (по умолчанию текущая): ").strip()
        save_dir = save_dir if save_dir else '.'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'pretrained_model.h5')
        scaler_X_path = os.path.join(save_dir, 'scaler_X.save')
        scaler_y_path = os.path.join(save_dir, 'scaler_y.save')
        predictor.model.save(model_path)
        joblib.dump(predictor.scaler_X, scaler_X_path)
        joblib.dump(predictor.scaler_y, scaler_y_path)
        print(f"Модель и scaler'ы сохранены в {save_dir}")
        
    elif choice == '2':
        model_path = input("Введите путь к сохраненной модели (.h5): ").strip()
        model_dir = os.path.dirname(model_path) or '.'
        scaler_X_path = os.path.join(model_dir, 'scaler_X.save')
        scaler_y_path = os.path.join(model_dir, 'scaler_y.save')
        predictor.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
            }
        )
        
        if not os.path.exists(model_path):
            print(f"Ошибка: файл модели {model_path} не найден.")
            return
        if not os.path.exists(scaler_X_path):
            print(f"Ошибка: файл scaler_X {scaler_X_path} не найден.")
            return
        if not os.path.exists(scaler_y_path):
            print(f"Ошибка: файл scaler_y {scaler_y_path} не найден.")
            return
        
        predictor.model = tf.keras.models.load_model(model_path)
        predictor.scaler_X = joblib.load(scaler_X_path)
        predictor.scaler_y = joblib.load(scaler_y_path)
        print("Модель и scaler'ы успешно загружены.")
    else:
        print("Неверный выбор. Выходим.")
        return
    
    last_year_data = X[-1]
    predictions = predictor.predict_next_year(last_year_data)
    months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
    
    print("\nПрогноз на следующий год (температура по месяцам):")
    for month, temp in zip(months, predictions):
        print(f"{month}: {temp:.1f}°C")
    
    plot_predictions(predictions)
    if choice == '1':
        plot_training_history(predictor.history)
    np.savetxt('predictions.txt', predictions, fmt='%.1f')

if __name__ == "__main__":
    main()