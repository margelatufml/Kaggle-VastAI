from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

# 1. Settings
MODEL_NAME = "roberta-large"  # Or 'deberta-v3-large' for even better results if available
DATA_FILE = "unlabeled_pretraining_corpus.txt"
BATCH_SIZE = 16  # Lower if OOM

# 2. Load text dataset
dataset = load_dataset('text', data_files=DATA_FILE)

# 3. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128)
tokenized = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

# 4. Data Collator for MLM
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# 5. Model
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# 6. Training Args
args = TrainingArguments(
    output_dir="./mlm_spooky",
    overwrite_output_dir=True,
    num_train_epochs=2,      # 1-2 is enough with 50k+ lines
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    save_total_limit=2,
    learning_rate=5e-5,
    prediction_loss_only=True,
    report_to="none",
    fp16=True,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    data_collator=collator,
)
trainer.train()

# **VERY IMPORTANT: Save both model and tokenizer**
trainer.save_model("./mlm_spooky")
tokenizer.save_pretrained("./mlm_spooky")  # <-- ADD THIS LINE

print("MLM Pretraining Complete. Model and tokenizer saved to ./mlm_spooky")
