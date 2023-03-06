import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

# Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Prompt the user to enter the question
question = input("Enter the question: ")

# Tokenize the input
inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors="pt")

# Get the start and end positions of the answer
start_positions, end_positions = model(**inputs).values()

# Move the tensors to CPU and flatten them to convert to Python lists
start_positions = torch.flatten(start_positions.cpu()).tolist()
end_positions = torch.flatten(end_positions.cpu()).tolist()

# Iterate over the start and end positions and extract the answer for each
answers = []
for start, end in zip(start_positions, end_positions):
    if start < end:
        start = int(start)
        end = int(end)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end+1]))
        answers.append(answer)

print(answers)
