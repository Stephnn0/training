from custom import data

print(data)

train_dataset = [{"text": f"<s>[INST] {item['question']} [/INST] {item['answer']} </s>"} for item in data]

print(train_dataset)
