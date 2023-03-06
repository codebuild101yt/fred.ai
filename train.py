from transformers import BertTokenizer, BertLMHeadModel, DataCollatorForLanguageModeling, LineByLineTextDataset, TrainingArguments, Trainer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='./train-v1.1.json',
    block_size=128
)

model = BertLMHeadModel.from_pretrained('bert-base-uncased')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=2e-5,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained('fred')
tokenizer.save_pretrained('fred')
