import tensorflow as tf
import numpy as np

# Create a basic model for y = 3x + 1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='adam', loss='mse')

# Train on sample data
x_train = np.array([0, 1, 2, 3, 4], dtype=float)
y_train = 3 * x_train + 1
model.fit(x_train, y_train, epochs=200, verbose=0)

# Real-time prediction loop
while True:
    try:
        x = float(input("Input x → "))
        y = model.predict([[x]], verbose=0)[0][0]
        print(f"Predicted y: {y:.2f}\n")
    except:
        print("Invalid input. Type a number.\n")
