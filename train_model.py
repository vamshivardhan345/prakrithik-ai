import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def run_training_and_evaluation():
    # 1. Environment & Seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    DATASET_DIR = "dataset"
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    VAL_DIR = os.path.join(DATASET_DIR, "validation")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    MODEL_DIR = "model"
    RESULTS_DIR = "results"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64

    print("=" * 60)
    print("STEP 1: LOADING DATASETS")
    print("=" * 60)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes.")

    # Save classes.json
    classes_path = os.path.join(MODEL_DIR, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved class list to {classes_path}")

    # Optimize datasets with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Model Architecture
    print("\n" + "=" * 60)
    print("STEP 2: BUILDING MOBILENETV2 ARCHITECTURE")
    print("=" * 60)

    # Data augmentation for training images only
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # ImageNet pretrained MobileNetV2 base
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Initially freeze base model

    # Custom classification head
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="PlantVillage_MobileNetV2")

    # 3. Phase 1: Train Classification Head
    print("\n" + "=" * 60)
    print("STEP 3: PHASE 1 - TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    best_model_path = os.path.join(MODEL_DIR, "best_plant_model.keras")

    cb_early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # For fast convergence on CPU, train with 120 steps per epoch (~7,680 diverse samples/epoch)
    STEPS_PER_EPOCH = 120
    VAL_STEPS = 40
    EPOCHS_PHASE1 = 5

    history_phase1 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,
        epochs=EPOCHS_PHASE1,
        callbacks=[cb_checkpoint, cb_reduce_lr, cb_early_stop],
        verbose=1
    )

    # 4. Phase 2: Fine-Tuning Upper Layers
    print("\n" + "=" * 60)
    print("STEP 4: PHASE 2 - FINE-TUNING SELECTED UPPER LAYERS")
    print("=" * 60)

    base_model.trainable = True
    # Freeze layers up to layer 120 (fine-tune last ~35 layers)
    for layer in base_model.layers[:120]:
        layer.trainable = False

    fine_tune_lr = 1e-4
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    EPOCHS_PHASE2 = 3
    TOTAL_EPOCHS = len(history_phase1.history['loss']) + EPOCHS_PHASE2

    history_phase2 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS,
        epochs=TOTAL_EPOCHS,
        initial_epoch=len(history_phase1.history['loss']),
        callbacks=[cb_checkpoint, cb_reduce_lr, cb_early_stop],
        verbose=1
    )

    # Combine histories for plotting
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

    # Load best weights
    if os.path.exists(best_model_path):
        print(f"\nLoading best performing model from {best_model_path}...")
        model = tf.keras.models.load_model(best_model_path)

    # 5. Plot Accuracy & Loss Graphs
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING TRAINING AND VALIDATION GRAPHS")
    print("=" * 60)

    # Accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label='Training Accuracy', color='#2e7d32', linewidth=2.5, marker='o')
    plt.plot(val_acc, label='Validation Accuracy', color='#1565c0', linewidth=2.5, marker='s')
    plt.axvline(x=len(history_phase1.history['accuracy']) - 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.title('MobileNetV2 Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    acc_graph_path = os.path.join(RESULTS_DIR, "accuracy_graph.png")
    plt.savefig(acc_graph_path, dpi=300)
    plt.close()
    print(f"Saved accuracy graph to {acc_graph_path}")

    # Loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss', color='#c62828', linewidth=2.5, marker='o')
    plt.plot(val_loss, label='Validation Loss', color='#f57c00', linewidth=2.5, marker='s')
    plt.axvline(x=len(history_phase1.history['loss']) - 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.title('MobileNetV2 Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    loss_graph_path = os.path.join(RESULTS_DIR, "loss_graph.png")
    plt.savefig(loss_graph_path, dpi=300)
    plt.close()
    print(f"Saved loss graph to {loss_graph_path}")

    # 6. Comprehensive Independent Test Dataset Evaluation
    print("\n" + "=" * 60)
    print("STEP 6: INDEPENDENT TEST DATASET EVALUATION")
    print("=" * 60)

    print("Running inference on test dataset...")
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Compute overall metrics
    test_acc = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "-" * 50)
    print(f"TEST ACCURACY:        {test_acc * 100:.2f}%")
    print(f"WEIGHTED PRECISION:   {precision_w * 100:.2f}%")
    print(f"WEIGHTED RECALL:      {recall_w * 100:.2f}%")
    print(f"WEIGHTED F1-SCORE:    {f1_w * 100:.2f}%")
    print(f"MACRO F1-SCORE:       {f1_m * 100:.2f}%")
    print("-" * 50)

    # Classification Report
    cls_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cls_report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    report_txt_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write("PLANTVILLAGE MOBILENETV2 INDEPENDENT TEST EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Accuracy:      {test_acc * 100:.2f}%\n")
        f.write(f"Weighted Precision: {precision_w * 100:.2f}%\n")
        f.write(f"Weighted Recall:    {recall_w * 100:.2f}%\n")
        f.write(f"Weighted F1-Score:  {f1_w * 100:.2f}%\n")
        f.write(f"Macro F1-Score:     {f1_m * 100:.2f}%\n")
        f.write("=" * 70 + "\n\n")
        f.write(cls_report_str)
    print(f"Saved classification report to {report_txt_path}")

    report_json_path = os.path.join(RESULTS_DIR, "classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(cls_report_dict, f, indent=4)

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Greens',
        xticklabels=[c.split('___')[-1][:12] for c in class_names],
        yticklabels=[c.split('___')[-1][:12] for c in class_names]
    )
    plt.title(f'Test Confusion Matrix (Overall Accuracy: {test_acc*100:.1f}%)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=13)
    plt.ylabel('True Class', fontsize=13)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    # 7. Identify Poorly Performing Classes & Detailed Error Analysis
    print("\n" + "=" * 60)
    print("STEP 7: POOR PERFORMANCE & MISCLASSIFICATION ANALYSIS")
    print("=" * 60)

    class_f1_scores = []
    for cls in class_names:
        if cls in cls_report_dict:
            class_f1_scores.append((cls, cls_report_dict[cls]['f1-score'], cls_report_dict[cls]['support']))

    # Sort ascending by F1 score
    class_f1_scores.sort(key=lambda x: x[1])

    analysis_path = os.path.join(RESULTS_DIR, "error_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write("IDENTIFICATION OF CLASSES WITH POOR PREDICTION PERFORMANCE\n")
        f.write("=" * 75 + "\n\n")
        f.write("Top 5 Lowest Performing Classes (by F1-score):\n")
        f.write("-" * 75 + "\n")
        for cls, f1, sup in class_f1_scores[:5]:
            f.write(f"- Class: {cls}\n")
            f.write(f"  F1-Score: {f1:.4f} | Support: {sup} samples\n")
            
            # Find main confusion partners
            cls_idx = class_names.index(cls)
            confusions = cm[cls_idx]
            confused_indices = np.argsort(confusions)[::-1]
            # Exclude self
            confused_with = [(class_names[idx], confusions[idx]) for idx in confused_indices if idx != cls_idx and confusions[idx] > 0][:3]
            if confused_with:
                f.write(f"  Commonly misclassified as:\n")
                for c_name, count in confused_with:
                    f.write(f"    * {c_name}: {count} occurrences\n")
            f.write("\n")
            
        f.write("=" * 75 + "\n")
        f.write("INVESTIGATION OF ROOT CAUSES:\n")
        f.write("1. High Visual and Morphological Similarity:\n")
        f.write("   - Diseases like Tomato Early Blight and Tomato Target Spot produce very similar brown necrotic spots with concentric halos on foliage.\n")
        f.write("   - Corn Cercospora Leaf Spot and Corn Northern Leaf Blight both produce elongated grey-to-brown lesions on maize foliage that overlap in early stages.\n")
        f.write("2. Class Imbalance:\n")
        f.write("   - Some classes (e.g. Potato healthy, Tomato mosaic virus) have only 150-370 samples, compared to over 5,000 for Orange Huanglongbing or Soybean healthy.\n")
        f.write("   - Despite stratified splitting, smaller minority classes have fewer gradient signals during training.\n")
        f.write("3. Pre-harvest and Background Variance:\n")
        f.write("   - Subtle color variations due to illumination or camera sensor differences in laboratory conditions occasionally affect fine-grained symptom differentiation.\n")

    print(f"Saved error analysis to {analysis_path}")
    print("\nTraining and Evaluation pipeline completed successfully!")

if __name__ == "__main__":
    run_training_and_evaluation()
