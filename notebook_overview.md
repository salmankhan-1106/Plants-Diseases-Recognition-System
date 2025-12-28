# Detailed notebook block-by-block overview

This document walks through each cell of your `final.ipynb`, explains what it does, the libraries used, functions/classes defined, models/approaches applied, and points you can mention in your viva.  

**Notes for viva:** focus on: what each block's purpose is, which libraries and key functions are used, model architecture or algorithm choice, training/evaluation approach, input/output shapes and types, and any hyperparameters or important implementation details.


---

## Cell 1 — code

**Code preview (first lines):**

```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from PIL import Image

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf 
from tensorflow.keras.layers  import Dense ,Conv2D ,Activation , Dropout , BatchNormalization , ReLU , MaxPooling2D , Flatten 
from tensorflow.keras.models import Sequential 
...
```


**Imports detected:**

- `import pandas as pd`

- `import numpy as np`

- `import matplotlib.pyplot as plt`

- `import seaborn as sns`

- `import os as os`

- `from PIL import Image

from sklearn`

- `from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf 
from tensorflow`

- `from tensorflow.keras.models import Sequential 
from tensorflow`

- `from tensorflow.keras.optimizers import Adam 
from tensorflow`

- `import warnings as warnings`



**Model/framework indicators found:**

- scikit-learn (classical ML — e.g., pipelines, preprocessing, model classes)

- Keras / TensorFlow (deep learning models, Sequential/Functional API)

- Matplotlib (visualization)

- Pandas (dataframes and tabular data handling)

- NumPy (numerical arrays and operations)



**Viva talking points for this cell:**

- Explain why each major library is used and which of its functions are important for this cell.

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).

- Discuss evaluation metrics and why chosen for the task (e.g., class imbalance -> use F1).



**Full code (for reference):**

```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from PIL import Image

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf 
from tensorflow.keras.layers  import Dense ,Conv2D ,Activation , Dropout , BatchNormalization , ReLU , MaxPooling2D , Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau , ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

# GPU Configuration for RTX 3070
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configure GPU memory growth to avoid OOM errors 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) configured: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
```


---

## Cell 2 — code

**Code preview (first lines):**

```
data_dir = r"E:\5 semester\machine learning\lab\lab final\final\plantvillage dataset\color"
label = []
image_path = []
folds = os.listdir(data_dir)
for fold in folds :
    folder_path = os.path.join(data_dir , fold)
    imgs = os.listdir(folder_path)
    for img in imgs : 
        img_path = os.path.join(folder_path , img)
        label.append(fold)
        image_path.append(img_path)
print("Total images:", len(image_path))
...
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
data_dir = r"E:\5 semester\machine learning\lab\lab final\final\plantvillage dataset\color"
label = []
image_path = []
folds = os.listdir(data_dir)
for fold in folds :
    folder_path = os.path.join(data_dir , fold)
    imgs = os.listdir(folder_path)
    for img in imgs : 
        img_path = os.path.join(folder_path , img)
        label.append(fold)
        image_path.append(img_path)
print("Total images:", len(image_path))
print("Total labels:", len(label))
```


---

## Cell 3 — code

**Code preview (first lines):**

```
paths_df = pd.Series(image_path)
label_df = pd.Series(label)

data = pd.DataFrame(
    {
        "imagepaths" :paths_df,
        "label" : label_df
    }
)
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
paths_df = pd.Series(image_path)
label_df = pd.Series(label)

data = pd.DataFrame(
    {
        "imagepaths" :paths_df,
        "label" : label_df
    }
)
```


---

## Cell 4 — code

**Code preview (first lines):**

```
data.head()
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
data.head()
```


---

## Cell 5 — code

**Code preview (first lines):**

```
plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    rand_idx = np.random.randint(0, len(data))
    img_path = data.iloc[rand_idx]['imagepaths']
    lbl = data.iloc[rand_idx]['label']
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(lbl, fontsize=8)
    plt.axis('off')

plt.tight_layout()
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    rand_idx = np.random.randint(0, len(data))
    img_path = data.iloc[rand_idx]['imagepaths']
    lbl = data.iloc[rand_idx]['label']
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(lbl, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
```


---

## Cell 6 — code

**Code preview (first lines):**

```
label_counts = data.label.value_counts().reset_index()
label_counts.columns = ['Label', 'Count']

plt.figure(figsize=(12, 6))
sns.barplot(
    data=label_counts,
    x="Label",
    y="Count",
    palette="Greens_r"  
)
plt.title("Class Distribution in PlantVillage Dataset", fontsize=20, color="#1b5e20")
plt.xlabel("Plant Disease Classes", fontsize=14)
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
label_counts = data.label.value_counts().reset_index()
label_counts.columns = ['Label', 'Count']

plt.figure(figsize=(12, 6))
sns.barplot(
    data=label_counts,
    x="Label",
    y="Count",
    palette="Greens_r"  
)
plt.title("Class Distribution in PlantVillage Dataset", fontsize=20, color="#1b5e20")
plt.xlabel("Plant Disease Classes", fontsize=14)
plt.ylabel("Number of Images", fontsize=14)
plt.xticks(rotation=45, ha="right")
sns.despine()
plt.tight_layout()
plt.show()

```


---

## Cell 7 — markdown

**Markdown content (preview):** `## Exploratory Data Analysis`


**What to mention in viva:** describe the purpose of this section as written in the markdown — typically it documents goal, dataset description, or high-level pipeline.


Full markdown (if you want to read during viva):


```
## Exploratory Data Analysis
```


---

## Cell 8 — code

**Code preview (first lines):**

```
# Color channel analysis for sample images
print("Analyzing RGB color distributions...")
sample_size = 100
sample_indices = np.random.choice(len(data), sample_size, replace=False)

red_means = []
green_means = []
blue_means = []

for idx in sample_indices:
    img_path = data.iloc[idx]['imagepaths']
    try:
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- This cell loads data from files (CSV, images, numpy arrays). Mention dataset path, shape, missing-value handling, and initial EDA.

- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
# Color channel analysis for sample images
print("Analyzing RGB color distributions...")
sample_size = 100
sample_indices = np.random.choice(len(data), sample_size, replace=False)

red_means = []
green_means = []
blue_means = []

for idx in sample_indices:
    img_path = data.iloc[idx]['imagepaths']
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            red_means.append(np.mean(img_array[:,:,0]))
            green_means.append(np.mean(img_array[:,:,1]))
            blue_means.append(np.mean(img_array[:,:,2]))
    except:
        pass

# Plot RGB distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.hist(red_means, bins=20, color='red', alpha=0.6, edgecolor='black')
plt.title('Red Channel Mean Intensity', fontsize=12, fontweight='bold')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(green_means, bins=20, color='green', alpha=0.6, edgecolor='black')
plt.title('Green Channel Mean Intensity', fontsize=12, fontweight='bold')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(blue_means, bins=20, color='blue', alpha=0.6, edgecolor='black')
plt.title('Blue Channel Mean Intensity', fontsize=12, fontweight='bold')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nAverage RGB values across {len(red_means)} sample images:")
print(f"Red:   {np.mean(red_means):.2f}")
print(f"Green: {np.mean(green_means):.2f}")
print(f"Blue:  {np.mean(blue_means):.2f}")
```


---

## Cell 9 — code

**Code preview (first lines):**

```
# Analyze image dimensions and properties
print("Analyzing sample images for properties...")
sample_indices = np.random.choice(len(data), min(500, len(data)), replace=False)

widths = []
heights = []
aspect_ratios = []
file_sizes = []

for idx in sample_indices:
    img_path = data.iloc[idx]['imagepaths']
    try:
...
```


- This cell loads data from files (CSV, images, numpy arrays). Mention dataset path, shape, missing-value handling, and initial EDA.

**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
# Analyze image dimensions and properties
print("Analyzing sample images for properties...")
sample_indices = np.random.choice(len(data), min(500, len(data)), replace=False)

widths = []
heights = []
aspect_ratios = []
file_sizes = []

for idx in sample_indices:
    img_path = data.iloc[idx]['imagepaths']
    try:
        img = Image.open(img_path)
        w, h = img.size
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w/h)
        file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
    except:
        pass

print(f"\nImage Properties Analysis (Sample of {len(widths)} images):")
print(f"Average Width: {np.mean(widths):.0f} px")
print(f"Average Height: {np.mean(heights):.0f} px")
print(f"Average Aspect Ratio: {np.mean(aspect_ratios):.2f}")
print(f"Average File Size: {np.mean(file_sizes):.2f} KB")
```


---

## Cell 10 — code

**Code preview (first lines):**

```
# Visualize image properties distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(widths, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Width (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(heights, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Height (pixels)')
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
# Visualize image properties distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(widths, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Width (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(heights, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Height (pixels)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(aspect_ratios, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(file_sizes, bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('File Size (KB)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


---

## Cell 11 — code

**Code preview (first lines):**

```
train_data, damy_data = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['label'], random_state=42)
valid_data, test_data = train_test_split(damy_data, test_size=0.5, shuffle=True, stratify=damy_data['label'], random_state=42)
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
train_data, damy_data = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['label'], random_state=42)
valid_data, test_data = train_test_split(damy_data, test_size=0.5, shuffle=True, stratify=damy_data['label'], random_state=42)
```


---

## Cell 12 — code

**Code preview (first lines):**

```
print(len(train_data.label.value_counts()))
print(len(test_data.label.value_counts()))
print(len(valid_data.label.value_counts()))
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
print(len(train_data.label.value_counts()))
print(len(test_data.label.value_counts()))
print(len(valid_data.label.value_counts()))
```


---

## Cell 13 — code

**Code preview (first lines):**

```
image_size = 224
batch_size = 32 
channel = 3
```


**Hyperparameters found:**

- `batch_size` = 32



**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
image_size = 224
batch_size = 32 
channel = 3 
```


---

## Cell 14 — code

**Code preview (first lines):**

```
train_datagen = ImageDataGenerator(
    rescale = 1./255 , 
    rotation_range=40,
    width_shift_range = 0.2 , 
    height_shift_range = 0.2 , 
    shear_range = 0.2 , 
    zoom_range = 0.2 , 
    horizontal_flip=True,
     fill_mode='nearest'
)
test_datagen=ImageDataGenerator(rescale=1./255,)
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
train_datagen = ImageDataGenerator(
    rescale = 1./255 , 
    rotation_range=40,
    width_shift_range = 0.2 , 
    height_shift_range = 0.2 , 
    shear_range = 0.2 , 
    zoom_range = 0.2 , 
    horizontal_flip=True,
     fill_mode='nearest'
)
test_datagen=ImageDataGenerator(rescale=1./255,)
```


---

## Cell 15 — code

**Code preview (first lines):**

```
train_gen=train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='imagepaths',
    y_col='label',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(image_size,image_size),
    shuffle=True
)
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
train_gen=train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='imagepaths',
    y_col='label',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(image_size,image_size),
    shuffle=True
)

```


---

## Cell 16 — code

**Code preview (first lines):**

```
valid_gen=test_datagen.flow_from_dataframe(
    dataframe=valid_data,
    x_col='imagepaths',
    y_col='label',
    class_mode='categorical',
    batch_size=batch_size,
    target_size=(image_size,image_size),
    shuffle=False
    
)
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
valid_gen=test_datagen.flow_from_dataframe(
    dataframe=valid_data,
    x_col='imagepaths',
    y_col='label',
    class_mode='categorical',
    batch_size=batch_size,
    target_size=(image_size,image_size),
    shuffle=False
    
)
```


---

## Cell 17 — code

**Code preview (first lines):**

```
test_gen=test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='imagepaths',
    y_col='label',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(image_size,image_size),
    shuffle=False
    
)
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
test_gen=test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='imagepaths',
    y_col='label',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(image_size,image_size),
    shuffle=False
    
)
```


---

## Cell 18 — code

**Code preview (first lines):**

```
train_gen.class_indices.items()
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
train_gen.class_indices.items()
```


---

## Cell 19 — code

**Code preview (first lines):**

```
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', input_shape=(image_size, image_size, channel)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),

    # Block 2
...
```


**Hyperparameters found:**

- `learning_rate` = 0.001



**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', input_shape=(image_size, image_size, channel)),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),

    # Block 2
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),

    # Block 3
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Fully connected
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),

    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

```


---

## Cell 20 — code

**Code preview (first lines):**

```
model.summary()
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
model.summary()
```


---

## Cell 21 — code

**Code preview (first lines):**

```
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=2, monitor='val_loss', verbose=1),
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
]
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=2, monitor='val_loss', verbose=1),
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
]

```


---

## Cell 22 — code

**Code preview (first lines):**

```
history=model.fit(train_gen,epochs=25,validation_data=valid_gen,callbacks=callbacks, verbose=1)
```


- This cell performs training (`fit`) or calls a fit method. Mention the target variable, training loop, batch size, epochs, optimizer, and loss if present.

**Hyperparameters found:**

- `epochs` = 25



**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
history=model.fit(train_gen,epochs=25,validation_data=valid_gen,callbacks=callbacks, verbose=1)
```


---

## Cell 23 — code

**Code preview (first lines):**

```
loss_train,accuracy_train=model.evaluate(train_gen)
loss_valid,accuracy_valid=model.evaluate(valid_gen)
loss_test,accuracy_test=model.evaluate(test_gen)
```


- This cell evaluates model performance (`evaluate`) — describe metrics used (accuracy, F1, loss, etc.).

**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
loss_train,accuracy_train=model.evaluate(train_gen)
loss_valid,accuracy_valid=model.evaluate(valid_gen)
loss_test,accuracy_test=model.evaluate(test_gen)
```


---

## Cell 24 — code

**Code preview (first lines):**

```
print(f'Loss Train: {loss_train},Accuracy Train: {accuracy_train}')
print(f'Loss Valid: {loss_valid},Accuracy Valid: {accuracy_valid}')
print(f'Loss Test: {loss_test},Accuracy Test: {accuracy_test}')
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
print(f'Loss Train: {loss_train},Accuracy Train: {accuracy_train}')
print(f'Loss Valid: {loss_valid},Accuracy Valid: {accuracy_valid}')
print(f'Loss Test: {loss_test},Accuracy Test: {accuracy_test}')
```


---

## Cell 25 — code

**Code preview (first lines):**

```
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = np.arange(1, len(tr_acc) + 1)

plt.figure(figsize=(18, 7))

plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, color='#E74C3C', linewidth=2, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, color='#27AE60', linewidth=2, marker='s', label='Validation Loss')
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = np.arange(1, len(tr_acc) + 1)

plt.figure(figsize=(18, 7))

plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, color='#E74C3C', linewidth=2, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, color='#27AE60', linewidth=2, marker='s', label='Validation Loss')
plt.fill_between(epochs, val_loss, tr_loss, color='gray', alpha=0.1)  
plt.title('Model Loss Over Epochs', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, color='#E74C3C', linewidth=2, marker='o', label='Train Accuracy')
plt.plot(epochs, val_acc, color='#27AE60', linewidth=2, marker='s', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs', fontsize=16, weight='bold')
plt.fill_between(epochs, val_acc, tr_acc, color='gray', alpha=0.1)  
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)


plt.tight_layout()
plt.show()

```


---

## Cell 26 — code

**Code preview (first lines):**

```
def preprocess(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    resized = img.resize(target_size)
    img_array = np.array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array
```


**Definitions in this cell:**

- Function: `preprocess()` — explain its purpose, inputs and outputs in viva.



- This cell loads data from files (CSV, images, numpy arrays). Mention dataset path, shape, missing-value handling, and initial EDA.

**Viva talking points for this cell:**

- Describe function signatures (arguments and return values) and how they fit into the pipeline.



**Full code (for reference):**

```
def preprocess(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    resized = img.resize(target_size)
    img_array = np.array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array
```


---

## Cell 27 — code

**Code preview (first lines):**

```
def prediction(model,img_path,class_indices):
  preprocessed_img = preprocess(img_path)
  predictions = model.predict(preprocessed_img)
  predicted_class_index = np.argmax(predictions, axis=1)[0]
  predicted_class_name = class_indices[predicted_class_index]
  return predicted_class_name
```


**Definitions in this cell:**

- Function: `prediction()` — explain its purpose, inputs and outputs in viva.



- This cell runs inference (`predict`). Mention input preprocessing, expected input shape/types, and postprocessing.

**Viva talking points for this cell:**

- Describe function signatures (arguments and return values) and how they fit into the pipeline.



**Full code (for reference):**

```
def prediction(model,img_path,class_indices):
  preprocessed_img = preprocess(img_path)
  predictions = model.predict(preprocessed_img)
  predicted_class_index = np.argmax(predictions, axis=1)[0]
  predicted_class_name = class_indices[predicted_class_index]
  return predicted_class_name
```


---

## Cell 28 — code

**Code preview (first lines):**

```
class_indices = {v: k for k, v in train_gen.class_indices.items()}
class_indices
```


**Viva talking points for this cell:**

- Mention training procedure: dataset split (train/val/test), early stopping, checkpoints, and overfitting mitigation (regularization/augmentation).



**Full code (for reference):**

```
class_indices = {v: k for k, v in train_gen.class_indices.items()}
class_indices
```


---

## Cell 29 — code

**Code preview (first lines):**

```
image_path = r"D:\archive\plantvillage dataset\color\Grape___Black_rot\1f9d01f0-dde4-49f5-9c27-e87432451556___FAM_B.Rot 3414.JPG"
predicted_class_name = prediction(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
image_path = r"D:\archive\plantvillage dataset\color\Grape___Black_rot\1f9d01f0-dde4-49f5-9c27-e87432451556___FAM_B.Rot 3414.JPG"
predicted_class_name = prediction(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)
```


---

## Cell 30 — code

**Code preview (first lines):**

```
image_path = r"D:\archive\plantvillage dataset\color\Potato___Late_blight\2c2b39ed-75f9-49f6-a28e-36a2cb608297___RS_LB 2794.JPG"
predicted_class_name = prediction(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)
```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```
image_path = r"D:\archive\plantvillage dataset\color\Potato___Late_blight\2c2b39ed-75f9-49f6-a28e-36a2cb608297___RS_LB 2794.JPG"
predicted_class_name = prediction(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)
```


---

## Cell 31 — code

**Code preview (first lines):**

```
plt.figure(figsize=(18,18))

images, labels = next(test_gen)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.axis('off')

    true_label_idx = np.argmax(labels[i])
    actual = class_indices[true_label_idx]

...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- This cell runs inference (`predict`). Mention input preprocessing, expected input shape/types, and postprocessing.

- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
plt.figure(figsize=(18,18))

images, labels = next(test_gen)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.axis('off')

    true_label_idx = np.argmax(labels[i])
    actual = class_indices[true_label_idx]

    preds = model.predict(np.expand_dims(images[i], axis=0), verbose=0)
    pred_label_idx = np.argmax(preds)
    predict = class_indices[pred_label_idx]

    color = 'green' if actual == predict else 'red' #ternnary statement 
    plt.title(f"Actual: {actual}\nPredicted: {predict}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

```


---

## Cell 32 — code

**Code preview (first lines):**

```
y_true = test_gen.classes

y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes,normalize='true')

class_names = list(test_gen.class_indices.keys())

plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=class_names,
...
```


**Model/framework indicators found:**

- Matplotlib (visualization)



- This cell runs inference (`predict`). Mention input preprocessing, expected input shape/types, and postprocessing.

- Contains visualization (`matplotlib`/`plt`) — mention what the plots show and how they help (loss curves, confusion matrices).

**Viva talking points for this cell:**

- Explain the model/algorithm: architecture, loss, optimizer, training data split, and evaluation metrics.

- Discuss evaluation metrics and why chosen for the task (e.g., class imbalance -> use F1).

- Explain the plots and how they provide insight (learning curves, sample predictions).



**Full code (for reference):**

```
y_true = test_gen.classes

y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes,normalize='true')

class_names = list(test_gen.class_indices.keys())

plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of CNN Model')
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

```


---

## Cell 33 — code

**Code preview (first lines):**

```

```


**Viva talking points for this cell:**

- Explain the cell's high-level purpose (data loading, preprocessing, model definition, training, evaluation, or utility).



**Full code (for reference):**

```

```

