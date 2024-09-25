import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, GRU # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
 
# Definir o alfabeto permitido nos captchas (A-Z, 0-9, etc.)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
num_classes = len(alphabet) + 1  # +1 para o caractere em branco da CTC
 
# 1. Função para pré-processamento de imagem
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 64))  # Redimensionar para 128x64
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # Binarização
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Remover ruído
    return image / 255.0  # Normalizar para [0, 1]
 
# 2. Função para data augmentation
def augment_image(image):
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.Noise(p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    return augmented['image'].numpy()
 
# 3. Função para converter string de texto para vetor numérico (label)
def text_to_labels(text):
    return [alphabet.index(char) for char in text]
 
# 4. Função para carregar e processar os dados
def load_data(image_dir, csv_file):
    df = pd.read_csv(csv_file)  # Carregar o CSV
    images = []
    labels = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image_name'])
        image = preprocess_image(image_path)  # Pré-processar a imagem
        label = text_to_labels(row['text'])  # Converter texto para labels numéricos
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)
 
# 5. Construir o modelo CRNN
def build_crnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Reshape(target_shape=(-1, 64))(x)  # Reshape para RNN
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
 
# 6. Função de perda CTC
def ctc_loss(y_true, y_pred):
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
 
# 7. Pipeline de Treinamento
def train_crnn(image_dir, csv_file, epochs=50, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, patience=10, batch_size=32):
    # Carregar dados
    images, labels = load_data(image_dir, csv_file)
    # Dividir dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Preparar dados para entrada no modelo (reshape)
    X_train = np.expand_dims(X_train, axis=-1)  # Adicionar canal de cor (grayscale)
    X_val = np.expand_dims(X_val, axis=-1)
    # Configurar hiperparâmetros
    input_shape = (64, 128, 1)  # Tamanho da imagem (64x128 grayscale)
    # Criar o modelo CRNN
    model = build_crnn_model(input_shape, num_classes)
    # Compilar o modelo
    model.compile(optimizer='adam', loss=ctc_loss)
    # Definir callback de early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Treinar o modelo
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    # Salvar o modelo treinado
    model.save('crnn_model.h5')
 
    print("Treinamento concluído e modelo salvo como 'crnn_model.h5'.")
 
# 8. Exemplo de execução
if __name__ == '__main__':
    # Definir o caminho para as imagens e o arquivo CSV
    image_directory = 'captcha_solver\\datasets'
    csv_labels = 'captcha_solver\\datasets\\labels.csv'
    # Iniciar o treinamento do modelo
    train_crnn(image_directory, csv_labels, epochs=50, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, patience=10, batch_size=32)