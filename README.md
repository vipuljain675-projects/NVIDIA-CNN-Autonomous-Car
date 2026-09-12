# 🚗 Autonomous Self-Driving Car using NVIDIA CNN & Computer Vision

An end-to-end Deep Learning and Classical Computer Vision system for Autonomous Driving. This project implements NVIDIA's 2016 Behavioral Cloning architecture trained on Udacity simulator telemetry data, alongside an OpenCV-based lane detection and autonomous steering system.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [End-to-End Deep Learning Pipeline (`Autonomous_Self_Driving_Car.ipynb`)](#-end-to-end-deep-learning-pipeline)
  - [1. Data Collection & Multi-Camera Setup](#1-data-collection--multi-camera-setup)
  - [2. The ±0.15 Steering Recovery Math](#2-the-015-steering-recovery-math)
  - [3. Data Exploration & Bias Balancing](#3-data-exploration--bias-balancing)
  - [4. Data Augmentation Pipeline](#4-data-augmentation-pipeline)
  - [5. Image Preprocessing & Normalization](#5-image-preprocessing--normalization)
  - [6. Real-Time Memory Batch Generator (`yield`)](#6-real-time-memory-batch-generator-yield)
  - [7. NVIDIA CNN Architecture Breakdown](#7-nvidia-cnn-architecture-breakdown)
  - [8. Forward Pass & Mathematical Intuition](#8-forward-pass--mathematical-intuition)
  - [9. Model Training & Optimization](#9-model-training--optimization)
  - [10. Evaluation Metrics: Regression Loss vs Accuracy](#10-evaluation-metrics-regression-loss-vs-accuracy)
- [Real-Time Inference Server (`drive.py`)](#-real-time-inference-server-drivepy)
- [Classical Computer Vision Pipeline (`OpenCV Lane Detection`)](#-classical-computer-vision-pipeline-opencv-lane-detection)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Research References](#-research-references)

---

## 🔍 Executive Overview

Autonomous driving requires real-time perception and decision making. This project tackles the problem through two complementary approaches:
1. **Deep Learning Behavioral Cloning**: Learning end-to-end steering predictions directly from raw camera pixels using a Convolutional Neural Network (inspired by NVIDIA's landmark 2016 paper *[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)*).
2. **Classical Edge & Lane Detection**: Utilizing OpenCV edge detection (Canny), Region of Interest (ROI) filtering, and Hough Line Transforms to detect lane boundaries and compute steering commands mathematically.

---

## 🏗️ System Architecture & Workflow

```
[ Udacity Simulator ] 
       │ (Sends 3-camera Images + Telemetry via WebSocket)
       ▼
[ WebSocket Server (Flask + python-socketio) ]
       │
       ▼
[ Image Preprocessing: Crop (60:135) ➔ YUV Conversion ➔ Gaussian Blur ➔ Resize (200x66) ➔ Normalize (/255.0) ]
       │
       ▼
[ Trained NVIDIA CNN Model (`model.h5`) ] ➔ [ Predicted Steering Angle: -1.0 to +1.0 ]
       │
       ▼
[ PID / Throttle Controller (Dynamic Speed Control) ]
       │
       ▼ (Transmits 'steer' and 'throttle' commands back to Simulator)
[ Udacity Simulator Autonomous Steering ]
```

---

## 🧠 End-to-End Deep Learning Pipeline

All model training, data exploration, augmentation, and neural network compilation are executed in `Autonomous_Self_Driving_Car.ipynb`.

### 1. Data Collection & Multi-Camera Setup

Driving data is captured using the Udacity Self-Driving Car Simulator in **Training Mode**. As a human drives the car around the track, the simulator logs telemetry at 10Hz:
- **Images**: Saved from 3 simulated dashboard cameras:
  - `Center Camera` (frontal view)
  - `Left Camera` (offset to the left side of the hood)
  - `Right Camera` (offset to the right side of the hood)
- **`driving_log.csv`**: Contains timestamps, filepaths for center/left/right images, `steering angle` ($-1.0$ to $+1.0$), `throttle`, `reverse`, and `speed`.

```python
# Loading and viewing driving telemetry
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv('driving_log.csv', names=columns)
```

---

### 2. The ±0.15 Steering Recovery Math

One of the most critical challenges in behavioral cloning is **accumulated error**: if a model is only trained on center camera images of flawless human driving, it never learns how to recover when it drifts toward the road edges.

To teach the model autonomous recovery without deliberately crashing during data collection, we utilize NVIDIA's multi-camera geometric compensation:

$$\text{Steering}_{\text{center}} = \theta_{\text{human}}$$
$$\text{Steering}_{\text{left}} = \theta_{\text{human}} + 0.15$$
$$\text{Steering}_{\text{right}} = \theta_{\text{human}} - 0.15$$

```python
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        
        # Center Camera: Unmodified human steering angle
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        
        # Left Camera: Offset to steer right (+0.15) back toward center
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.15)
        
        # Right Camera: Offset to steer left (-0.15) back toward center
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.15)
        
    return np.asarray(image_path), np.asarray(steering)
```

**Why this works**:
- The **Left Camera** sees the road shifted to the right. By pairing this image with an artificially increased steering angle ($+0.15$), the model learns: *"If the car drifts left, steer right to recenter."*
- The **Right Camera** sees the road shifted to the left. By pairing this image with a decreased steering angle ($-0.15$), the model learns: *"If the car drifts right, steer left to recenter."*
- **Result**: Triples dataset size ($3\times$) and builds self-correcting behavior.

---

### 3. Data Exploration & Bias Balancing

In human driving datasets, driving straight ($0.0^\circ$ steering angle) constitutes over 80% of all recorded frames. If an AI is trained on raw unbalanced data, it develops a severe straight-line bias and fails to turn at sharp corners.

```python
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)

# Flatten the distribution peak at 0.0
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]  # Discard samples exceeding 400 per bin
    remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)
```

**Impact**: Caps the maximum samples per steering bucket to $400$, ensuring sharp turns have equal representation relative to straight roads.

---

### 4. Data Augmentation Pipeline

To prevent overfitting and allow the model to generalize across varied lighting conditions, shadows, and slopes, we apply random geometric and photometric transformations on the fly using `imgaug`:

| Augmentation | Function | Description |
| :--- | :--- | :--- |
| **Zoom** | `zoom(image)` | Random scale magnification between $1.0\times$ and $1.3\times$ |
| **Pan** | `pan(image)` | Random translational shift along X and Y axes ($\pm 10\%$) |
| **Brightness** | `img_random_brightness(image)` | Multiplies HSV value channel by $0.2 - 1.2$ to simulate shadows & sunset |
| **Horizontal Flip** | `img_random_flip(image, angle)` | Horizontally flips image (`cv2.flip(image, 1)`) and inverts steering angle ($\theta = -\theta$) |

```python
def random_augment(image_path, steering_angle):
    image = mpimg.imread(image_path)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle
```

---

### 5. Image Preprocessing & Normalization

Raw camera frames ($160 \times 320 \times 3$) contain irrelevant visual noise (sky, trees, horizon, and the car's hood). The preprocessing pipeline optimizes the input for the NVIDIA CNN:

```python
def img_preprocess(img):
    # 1. Crop: Remove sky (top 60px) and car hood (bottom 25px)
    img = img[60:135, :, :]
    
    # 2. Color Space: Convert RGB to YUV (as specified by NVIDIA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # 3. Gaussian Blur: Filter high-frequency noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 4. Resize: Scale to NVIDIA standard input dimensions (200x66)
    img = cv2.resize(img, (200, 66))
    
    # 5. Min-Max Normalization: Scale pixel values to [0.0, 1.0]
    img = img / 255.0
    return img
```

- **Why YUV Color Space?** NVIDIA demonstrated that the YUV color space separates luminance ($Y$) from chrominance ($U, V$). This allows the network to distinguish lane lines even in intense glare or shadow conditions where RGB fails.

---

### 6. Real-Time Memory Batch Generator (`yield`)

Loading tens of thousands of augmented high-resolution images simultaneously into system RAM causes memory overflow (`OOM`). We use Python generators (`yield`) to generate mini-batches on-the-fly:

```python
def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))
```

- **Training**: Augmentations applied randomly $\to$ infinite unique training variations.
- **Validation**: Raw images preprocessed without distortion to measure true generalization.

---

### 7. NVIDIA CNN Architecture Breakdown

The neural network is implemented using TensorFlow/Keras following the 9-layer NVIDIA self-driving architecture:

```python
def nvidia_model():
    model = Sequential([
        # Layer 1: 24 filters, 5x5 kernel, 2x2 stride
        Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'),
        
        # Layer 2: 36 filters, 5x5 kernel, 2x2 stride
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        
        # Layer 3: 48 filters, 5x5 kernel, 2x2 stride
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        
        # Layer 4: 64 filters, 3x3 kernel, 1x1 stride
        Conv2D(64, (3, 3), activation='elu'),
        
        # Layer 5: 64 filters, 3x3 kernel, 1x1 stride
        Conv2D(64, (3, 3), activation='elu'),
        
        # Flattening 3D feature maps to 1D vector
        Flatten(),
        
        # Fully Connected Layers
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        
        # Final Output Layer: Single Continuous Value (Steering Angle)
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model
```

#### Layer-by-Layer Specifications:

| Layer Type | Filters / Units | Kernel Size | Stride | Output Shape | Activation | Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | - | - | - | $(66, 200, 3)$ | - | Normalized YUV Image |
| **Conv2D_1** | $24$ | $5 \times 5$ | $(2, 2)$ | $(31, 98, 24)$ | ELU | Low-level feature extraction (edges, gradients) |
| **Conv2D_2** | $36$ | $5 \times 5$ | $(2, 2)$ | $(14, 47, 36)$ | ELU | Mid-level feature extraction (road contours, curb lines) |
| **Conv2D_3** | $48$ | $5 \times 5$ | $(2, 2)$ | $(5, 22, 48)$ | ELU | Higher spatial pattern representation |
| **Conv2D_4** | $64$ | $3 \times 3$ | $(1, 1)$ | $(3, 20, 64)$ | ELU | Complex shape combination & lane boundary detection |
| **Conv2D_5** | $64$ | $3 \times 3$ | $(1, 1)$ | $(1, 18, 64)$ | ELU | Deep geometric road topology representation |
| **Flatten** | - | - | - | $(1152)$ | - | Converts 3D feature matrix into 1D linear array |
| **Dense_1** | $100$ | - | - | $(100)$ | ELU | High-dimensional steering control mapping |
| **Dense_2** | $50$ | - | - | $(50)$ | ELU | Decision boundary refinement |
| **Dense_3** | $10$ | - | - | $(10)$ | ELU | Final feature aggregation |
| **Output** | $1$ | - | - | $(1)$ | Linear | Continuous predicted steering angle ($\theta$) |

- **Why ELU (Exponential Linear Unit)?** Unlike ReLU, ELU does not suffer from the "Dying ReLU" problem and maintains non-zero smooth gradients for negative values, yielding smoother vehicle steering curves.

---

### 8. Forward Pass & Mathematical Intuition

```
[ Input: Preprocessed Image (39,600 values: 66x200x3) ]
                           │
                           ▼
          [ 5x5 Conv Filters slide across image ]
          (Dot product of weights & pixel patches)
                           │
                           ▼
           [ 5 Convolutional Feature Maps ]
         (Detects: lane lines, road edges, curves)
                           │
                           ▼
       [ Flatten: 3D Grid ➔ 1D Vector (1,152 values) ]
                           │
                           ▼
        [ Dense(100) ➔ Dense(50) ➔ Dense(10) ]
        (Progressively condenses visual features)
                           │
                           ▼
     [ Dense(1): Single Output Neuron = Steering Angle ]
               e.g. Steering = -0.23 (Left Turn)
```

---

### 9. Model Training & Optimization

The network is optimized using the **Adam Optimizer** with **Mean Squared Error (MSE)** loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)^2$$

Where $y_i$ is the actual human steering angle and $\hat{y}_i$ is the model's prediction.

```python
history = model.fit(
    batch_generator(X_train, y_train, batch_size=100, istraining=True),
    steps_per_epoch=300,
    epochs=10,
    validation_data=batch_generator(X_valid, y_valid, batch_size=100, istraining=False),
    validation_steps=200
)

# Persisting trained weights
model.save('model.h5')
```

---

### 10. Evaluation Metrics: Regression Loss vs Accuracy

> [!NOTE]
> **Why is there no "Percentage Accuracy" metric?**
> Autonomous steering prediction is a **Continuous Regression Problem**, not a Discrete Classification problem (like Cat vs Dog). If the actual steering angle is $-0.234$ and the model predicts $-0.231$, in classification this would be counted as "wrong (0%)", but in driving physics, this is an exceptionally precise prediction.

#### Training Results:
- **Final Training Loss (MSE)**: `0.0396`
- **Final Validation Loss (MSE)**: `0.0303`
- **Root Mean Squared Error (RMSE)**: $\sqrt{0.0303} \approx 0.174$ (Mean prediction error is only $\pm 0.17$ on a $[-1.0, +1.0]$ steering scale).
- **Physical Validation**: Zero track departures or collisions over continuous full-lap autonomous runs on Track 1 of Udacity Simulator.

---

## ⚡ Real-Time Inference Server (`drive.py`)

`drive.py` bridges the trained neural network with the Udacity Simulator in real time using an eventlet WebSocket server:

```python
@sio.on('telemetry')
def telemetry(sid, data):
    # 1. Extract speed and base64 encoded camera frame from simulator
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    
    # 2. Preprocess frame matching training pipeline
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    
    # 3. Model Inference (predict steering angle)
    steering_angle = float(model.predict(image, verbose=0)[0][0])
    
    # 4. Dynamic Throttle Calculation based on speed limit
    throttle = 1.0 - speed / speed_limit
    
    # 5. Transmit control packets back to simulator
    send_control(steering_angle, throttle)
```

---

## 📷 Classical Computer Vision Pipeline (`OpenCV Lane Detection`)

In addition to deep learning behavioral cloning, `trackmania_self_driving_car_openCV/` contains an algorithmic Computer Vision implementation:

```
[ Screen Grab / Video Feed ]
             │
             ▼
   [ Grayscale Conversion ]
             │
             ▼
 [ Gaussian Blur (7x7 Filter) ]
             │
             ▼
[ Canny Edge Detection (270, 350) ]
             │
             ▼
[ Region of Interest (ROI) Polygon Masking ]
             │
             ▼
 [ Probabilistic Hough Line Transform ]
             │
             ▼
[ Slope Calculation: Positive (Right Lane) & Negative (Left Lane) ]
             │
             ▼
[ Keyboard / Hardware Control Automation ]
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Autonomous_Self_Driving_Cars.git
cd Autonomous_Self_Driving_Cars
```

### 2. Set Up Virtual Environment (Python 3.10 / 3.11 recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install tensorflow keras numpy opencv-python flask python-socketio==4.6.0 python-engineio==3.13.2 eventlet pillow
```

---

## 🚀 How to Run

### Running the Autonomous Driving System:

1. **Launch the Udacity Self-Driving Car Simulator**:
   - Download the simulator from [Udacity Self-Driving Car Sim](https://github.com/udacity/self-driving-car-sim).
   - Open the simulator and choose **Autonomous Mode**.
   
2. **Start the Telemetry Server**:
   ```bash
   ./venv/bin/python drive.py
   ```
   *(Or activate the virtual environment and run `python drive.py`)*

3. The simulator will automatically connect via WebSocket on port `4567`. Watch the car steer autonomously in real time!

---

## 📚 Research References

1. **Bojarski, M. et al. (2016)**. *End to End Learning for Self-Driving Cars*. NVIDIA Corporation. [arXiv:1604.07316](https://arxiv.org/abs/1604.07316).
2. **Canny, J. (1986)**. *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
3. **Udacity**. *Self-Driving Car Nanodegree Behavioral Cloning Benchmark*.

---

## 👨‍💻 Author
- **Vipul Jain** ([@vipuljain675](https://github.com/vipuljain675))
- Built with ❤️ using Python, TensorFlow, OpenCV, and Flask.
