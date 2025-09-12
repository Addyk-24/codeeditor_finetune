from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from evaluate import load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch


dataset = load_dataset("imdb")


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model.config)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


labels = dataset["train"]["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
print(class_weights)

# Example: Mapping string labels to integers
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
dataset = dataset.map(lambda x: {"label": label_mapping[x["text_label"]]})

# Verify the mapping
print(dataset["train"][0])


# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Keep only the classification head trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory for saving model checkpoints
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    fp16=True                        # Enable mixed precision for faster training
)

print(training_args)

# # Load a metric (F1-score in this case)
# metric = load("f1")

# # Define a custom compute_metrics function
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

# Start training
trainer.train()


# Evaluate the model
results = trainer.evaluate()
print(results)


# Generate predictions
predictions = trainer.predict(tokenized_datasets["test"])
predicted_labels = predictions.predictions.argmax(axis=-1)

# Classification report
print(classification_report(tokenized_datasets["test"]["label"], predicted_labels))

# Confusion matrix
cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot()



# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Define dummy input for tracing
dummy_input = torch.randint(0, 100, (1, 128))

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "bert_classification.onnx",
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Model exported to ONNX format")