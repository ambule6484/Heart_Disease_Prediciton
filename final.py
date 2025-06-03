import numpy as np  #Used for handling data efficiently.
import pandas as pd
import matplotlib.pyplot as plt  #Used for visualizing data through graphs.
import seaborn as sns
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #Provides machine learning models and tools for processing data.
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential #Used for building a deep learning model (CNN).
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.animation as animation  # For animation
import plotly.express as px  # For interactive plots

df = pd.read_csv("D:\Projects\Project_Heart_disease_Prediction\dataset.csv")  
# path of the csv file for access data

# Data exploration
print("Dataset Info:")
df.info() #info. about csv how many row,column
print("\nDataset Description:")
print(df.describe()) #generate descriptive statistics of the dataset.

# Visualizations
sns.countplot(x='target', data=df, palette='RdBu_r') # for create bar chart
plt.title('Target Distribution')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(12, 10)) #create new figure plot
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn') #to create heatmap visualization
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Convert categorical columns to dummy variables
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Features and target
X = dataset.drop(['target'], axis=1)
y = dataset['target']

# Reshape data for CNN (CNN expects 3D input: samples, timesteps, features)
X_cnn = X.values.astype('float32').reshape(X.shape[0], X.shape[1], 1)

# Split data for optimization (validation set)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models without the hybrid RNN-LSTM model
models = {
    "KNN": KNeighborsClassifier(n_neighbors=12),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "CNN": Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
}

# Define a function to compute common metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    auc = roc_auc_score(y_true, y_pred) * 100
    return accuracy, precision, recall, f1, auc

# Initialize plot for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 50)  # Number of epochs
ax.set_ylim(0, 100)  # Accuracy range
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy (%)')
line, = ax.plot([], [], label='Accuracy', color='b', lw=2)
ax.legend()

# Training and evaluation with animation
def animate_accuracy(model, X_cnn, y, epochs=50):
    accuracies = []  # List to store accuracy values for each epoch

    for epoch in tqdm(range(epochs), desc=f"Training {model.name}", ncols=100):
        model.fit(X_cnn, y, epochs=1, batch_size=32, verbose=0)  # Train for 1 epoch
        y_pred = (model.predict(X_cnn) > 0.5).astype(int).flatten()
        accuracy, _, _, _, _ = evaluate_model(y, y_pred)
        accuracies.append(accuracy)

        # Update plot for each epoch
        line.set_data(range(epoch + 1), accuracies)
        plt.pause(0.1)  # Pause to update the plot

    return accuracies

# Define the CNN model
cnn_model = models["CNN"]
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Perform the training and animation
accuracies = animate_accuracy(cnn_model, X_cnn, y, epochs=50)

# Show the final plot after the animation is done
plt.show()

# Interactive Plot with Plotly (Vertical Bar Plot)
fig = px.bar(summary, x='Model', y='Accuracy (%)', title="Model Accuracy Comparison for Heart Disease Prediction", 
             color='Accuracy (%)', color_continuous_scale='Viridis', labels={'Accuracy (%)': 'Accuracy (%)', 'Model': 'Model'}, orientation='v')
fig.show()