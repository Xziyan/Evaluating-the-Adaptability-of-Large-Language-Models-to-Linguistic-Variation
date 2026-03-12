import yaml

with open("example_prompt.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print(data.keys())         # dict_keys(['poetry', 'prose', 'encyclopedia', 'information', 'spoken'])
print(data["poetry"][0])   # {'input': 'Donc luttons...', 'output': '<root> ... </root>'}

