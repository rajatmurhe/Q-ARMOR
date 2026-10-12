# Maintained implementation: actual validation predictions -> game -> next epoch.
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
              loss="categorical_crossentropy", metrics=["accuracy"])
history_log, controller = train_ids(
    model, X_train_balanced, y_train_balanced_int, X_val, y_val,
    label_encoder.classes_.tolist(), epochs=10, batch_fraction=0.10,
    seed=42, output_dir=PROJECT_ROOT / "results/notebook",
    controller_options={"scale_step": 0.15, "max_scale": 2.5})
# Evaluate test once after validation-selected checkpoint restoration.
y_true = y_test
y_pred = predict_labels(model, X_test, class_labels)
test_results = metrics(y_true, y_pred, class_labels)
cm = np.asarray(test_results["confusion_matrix"])
report_dict = test_results["per_class"]
save_json(PROJECT_ROOT / "results/notebook/test.json", test_results)
