API_MODELS = {
    "Llama3": {
        "provider": "together.ai",
        "entry_point": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "max_tokens": 128000,
        "tokenizer" :"meta-llama/Llama-3.3-70B-Instruct" 
    },
    #"Qwen": {
       # "provider": "together.ai",
        #"entry_point": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        #"max_tokens": 8000

    #},
    "Qwen3": {
        "provider": "together.ai",
        "entry_point": "Qwen/Qwen3-235B-A22B-fp8-tput",
        "max_tokens": 32768,
        "tokenizer": "Qwen/Qwen3-235B-A22B-FP8"
    }, #a very big reasoning qwen model

    "Nemotron": {
        "provider": "together.ai",
        "entry_point": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "max_tokens": 32769,
        "tokenizer":"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    },
    "DeepSeek-V3": {
        "provider": "together.ai",
        "entry_point": "deepseek-ai/DeepSeek-V3",
        "max_tokens": 32769,
        "tokenizer": "deepseek-ai/DeepSeek-V3"
    },
    "DeepSeek-R1": {
        "provider": "together.ai",
        "entry_point": "deepseek-ai/DeepSeek-R1",
        "max_tokens": 32769,
        "tokenizer": "deepseek-ai/DeepSeek-R1"
    } #i had the distilled version before 

    #chatgpt faut passer via api de openai
}

PARAMETERS = {
    "temperature": 0,
    "seed": 42,
    "top_p": 1.0
}



# Prompts in English 

#simple prompt zero-shot,*with 0 definition, output format in json
SIMPLE_PROMPT_DEFAULT=  """Extract all named entities of the following types: PERS, LOC, ORG, TIME, PROD, EVENT.
Return the result as a valid JSON object.
For each entity, include:
- "entity": the exact substring from the text
- "start": the character index where the entity begins
- "end": the character index where the entity ends (exclusive)

Here is the input text:
"{text}"

Your output must follow this format exactly:
{
  "PERS": [
    {"entity": "EntityName", "start": start_index, "end": end_index},
    ...
  ],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}
Only include entities from the specified categories. If none, use empty lists.
"""

#simple prompt zero-shot *with simple definitions, and some clarification*, output format in jsont 
ZEROSHOT_PROMPT= """Extract all named entities from the following types:
- Pers (Person): Refers to individuals or named groups of people.
- Loc (Localisations): Specific places such as cities, countries, landmarks.
- Org (Organisations): Named companies, institutions, or associations.
- Time (Temporal expressions): Dates, time. 
- Prod (Products): Named products.
- Event (Events): Named events. 

Return the result as a valid JSON object.
For each entity, include:
- "entity": the exact substring from the text
- "start": the character index where the entity begins
- "end": the character index where the entity ends (exclusive)

Important:
Work at the **character level**, not token or word level.
Some named entities may be *nested* within others. 
You must extract **all valid entities**, even if they overlap or one is contained inside another.

Here is the input text:
"{text}"

Your output must follow this format exactly:
{
  "PERS": [
    {"entity": "EntityName", "start": start_index, "end": end_index},
    ...
  ],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}
Only include entities from the specified categories. If none, use empty lists."""

#simple prompt *with definitions* few shots *with 4 examples*, output format in json
SIMPLE_PROMPT_4SHOTS= """Extract all named entities from the following types:
- Pers (Person): Refers to individuals or named groups of people.
- Loc (Localisations): Specific places such as cities, countries, landmarks.
- Org (Organisations): Named companies, institutions, or associations.
- Time (Temporal expressions): Dates, time periods.
- Prod (Products): Named products, satellites, machines, etc.
- Event (Events): Named events.

Return the result as a valid JSON object.
For each entity, include:
- "entity": the exact substring from the text
- "start": the character index where the entity begins
- "end": the character index where the entity ends (exclusive)

Important:
Work at the **character level**, not token or word level.
Some named entities may be *nested* within others. 
You must extract **all valid entities**, even if they overlap or one is contained inside another.

Here is the input text:
"{text}"

Your output must follow this format exactly:
{
  "PERS": [
    {"entity": "EntityName", "start": start_index, "end": end_index},
    ...
  ],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}

Only include entities from the specified categories. If none, use empty lists.

Here are some examples:
---
Example 1:
Text: "Jiang Qing, et la bande des Quatre agite le mouvement contre les chaînes culturelles du passé : de nombreuses œuvres anciennes, livres, sculptures, bâtiments, etc."
Output:
{
  "PERS": [
    {"entity": "Jiang Qing", "start": 0, "end": 10},
    {"entity": "la bande des Quatre", "start": 15, "end": 35}
  ],
  "LOC": [],
  "ORG": [],
  "TIME": [],
  "PROD": [],
  "EVENT": []
}

---
Example 2:
Text: "ERVY-LE-CHATEL - 2ème jour du Marché du Livre à la salle des fêtes de 10 h à 18 h. Organisé par Les Ombelles."
Output:
{
  "PERS": [],
  "LOC": [
    {"entity": "ERVY-LE-CHATEL", "start": 0, "end": 16}
  ],
  "ORG": [
    {"entity": "Les Ombelles", "start": 97, "end": 110}
  ],
  "TIME": [
    {"entity": "2ème jour du Marché du Livre", "start": 19, "end": 51},
    {"entity": "de 10 h à 18 h", "start": 76, "end": 91}
  ],
  "PROD": [],
  "EVENT": [
    {"entity": "Marché du Livre", "start": 37, "end": 54}
  ]
}

---
Example 3:
Text: "La rentrée atmosphérique de Tiangong-1 intéresse particulièrement l'agence européenne, mais aussi les autres agences spatiales."
Output:
{
  "PERS": [],
  "LOC": [],
  "ORG": [],
  "TIME": [],
  "PROD": [
    {"entity": "Tiangong-1", "start": 30, "end": 40}
  ],
  "EVENT": []
}

---
Example 4: 
Text: "Le Tour de France 2024 est l'un des événements sportifs majeurs en Europe."
Output:{
  "PERS": [],
  "LOC": [
    {"entity": "France", "start": 11, "end": 17},
    {"entity": "Europe", "start": 68, "end": 74}
  ],
  "ORG": [],
  "TIME": [
    {"entity": "2024", "start": 19, "end": 23}
  ],
  "PROD": [],
  "EVENT": [
    {"entity": "Tour de France 2024", "start": 4, "end": 23}
  ]
}

---

"""


#CHAIN OF THOUGHT PROMPTS COMMENCENT ICI
#chain of thought prompt *with no definiions, 0 examples, output format in json
COT_SIMPLE_PROMPT = '''Extract all named entities of the following types: PERS, LOC, ORG, TIME, PROD, EVENT.

Instructions:
1. Read the text and extract all entities belonging to the above categories.
2. Some entities may overlap or be nested, return **all** valid entities.
3. For each entity:
   - Extract the exact substring from the text.
   - Calculate its "start" and "end" index (end is exclusive).
   - Make sure that slicing the input with these indices returns the exact entity string.
   - If the same entity appears multiple times, include each occurrence.
4. Work at the **character level**, not token or word level.
5. Return the final result as a valid JSON object with the following structure:

{
  "PERS": [
    {"entity": "EntityName", "start": start_index, "end": end_index},
    ...
  ],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}
Only include entities from the specified categories. If none, use empty lists.
Now, here is your input text:
"{text}"
Let's go step by step.
'''

#chain of thought prompt refined (hint of nestled entities) *with definitions, with 0 examples, output format in json
COT_ZEROSHOT_PROMPT = '''You are an expert linguistic annotator. Your task is to extract named entities from the text.

The categories of named entities are:
- Pers (Person): Individuals or named groups of people.
- Loc (Localisations): Specific places such as cities, countries, landmarks. 
- Org (Organisations): Named companies, institutions, or associations.
- Time (Temporal expressions): Dates, time periods.
- Prod (Products): Named products.
- Event (Events): Named events.

Instructions:
1. Read the text and extract all entities belonging to the above categories.
2. Some entities may overlap or be nested (e.g., “Tour de France 2024” is an EVENT, and “France” is a LOC inside it). Return **all** valid entities.
3. For each entity:
   - Extract the exact substring from the text.
   - Calculate its "start" and "end" character index (end is exclusive).
   - Make sure that slicing the input with these indices returns the exact entity string.
   - Ensure that `end - start == len(entity)` including punctuation or whitespace in the entity string.
   - If the same entity appears multiple times, include each occurrence.
   - Work at the **character level**, not token or word level.
4. Finally, return the final result as a valid JSON object with the exactly following structure:

{
  "PERS": [{"entity": "...", "start": ..., "end": ...}, ...],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}

Now, here is your input text:
"{text}"
Let's go step by step.
'''

#chain of thought prompt *with 4 examples (like complex), output format in json
COT_PROMPT_4SHOTS = '''You are an expert linguistic annotator. Your task is to extract named entities from the text.

The categories of named entities are:
- Pers (Person): Individuals or named groups of people.
- Loc (Localisations): Specific places such as cities, countries, landmarks.
- Org (Organisations): Named companies, institutions, or associations.
- Time (Temporal expressions): Dates, time periods.
- Prod (Products): Named products.
- Event (Events): Named events.

Instructions:
1. Read the text and extract all entities belonging to the above categories.
2. Some entities may overlap or be nested (e.g., “Tour de France 2024” is an EVENT, and “France” is a LOC inside it). Return **all** valid entities.
3. For each entity:
   - Extract the exact substring from the text.
   - Calculate its "start" and "end" index (end is exclusive).
   - Make sure that slicing the input with these indices returns the exact entity string.
   - If the same entity appears multiple times, include each occurrence.
4. Work at the **character level**, not token or word level.
5. Return the final result as a valid JSON object with the following structure:

{
  "PERS": [{"entity": "...", "start": ..., "end": ...}, ...],
  "LOC": [...],
  "ORG": [...],
  "TIME": [...],
  "PROD": [...],
  "EVENT": [...]
}
Here are some examples:
---
Example 1:
Text: "Jiang Qing, et la bande des Quatre agite le mouvement contre les chaînes culturelles du passé : de nombreuses œuvres anciennes, livres, sculptures, bâtiments, etc."
Output:
{
  "PERS": [
    {"entity": "Jiang Qing", "start": 0, "end": 10},
    {"entity": "la bande des Quatre", "start": 15, "end": 35}
  ],
  "LOC": [],
  "ORG": [],
  "TIME": [],
  "PROD": [],
  "EVENT": []
}

---
Example 2:
Text: "ERVY-LE-CHATEL - 2ème jour du Marché du Livre à la salle des fêtes de 10 h à 18 h. Organisé par Les Ombelles."
Output:
{
  "PERS": [],
  "LOC": [
    {"entity": "ERVY-LE-CHATEL", "start": 0, "end": 16}
  ],
  "ORG": [
    {"entity": "Les Ombelles", "start": 97, "end": 110}
  ],
  "TIME": [
    {"entity": "2ème jour du Marché du Livre", "start": 19, "end": 51},
    {"entity": "de 10 h à 18 h", "start": 76, "end": 91}
  ],
  "PROD": [],
  "EVENT": [
    {"entity": "Marché du Livre", "start": 37, "end": 54}
  ]
}

---
Example 3:
Text: "La rentrée atmosphérique de Tiangong-1 intéresse particulièrement l'agence européenne, mais aussi les autres agences spatiales."
Output:
{
  "PERS": [],
  "LOC": [],
  "ORG": [],
  "TIME": [],
  "PROD": [
    {"entity": "Tiangong-1", "start": 30, "end": 40}
  ],
  "EVENT": []
}

---
Example 4: 
Text: "Le Tour de France 2024 est l'un des événements sportifs majeurs en Europe."
Output:{
  "PERS": [],
  "LOC": [
    {"entity": "France", "start": 11, "end": 17},
    {"entity": "Europe", "start": 68, "end": 74}
  ],
  "ORG": [],
  "TIME": [
    {"entity": "2024", "start": 19, "end": 23}
  ],
  "PROD": [],
  "EVENT": [
    {"entity": "Tour de France 2024", "start": 4, "end": 23}
  ]
}

---
Now, here is your input text:
"{text}"
Let's go step by step.
'''


#INSERT BALISES VERSIONS
#simple prompt zero-shot *with simple definitions, and some clarification* output with balise<>
ZEROSHOT_PROMPT_BALISE = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.


Here is the input text:
{text}
"""
# 1 EXAMPLE HELPING LLM UNDERTANDING THE FORMAT
ONESHOT_PROMPT_BALISE = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.
- Return ONLY valid XML — no extra comments, headers, or explanation.


Here is an example:
Input:
"Paris est la capitale de la France."

Output:
<root>
<entity type="LOC">Paris</entity> est la capitale de la <entity type="LOC">France</entity>.
</root>

Now process the following input:
{text}
"""
#few shot, including nesteld entities example
FEWSHOT_PROMPT_BALISE = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.
- Return ONLY valid XML — no extra comments, headers, or explanation.


Here are some examples:
Input:
"Pendant l'exposition universelle de 1889 à Paris, organisée au Champ-de-Mars, Gustave Eiffel a présenté la tour Eiffel, construite par la société SETE."

Output:
<root>
Pendant l'<entity type="EVENT">exposition universelle de <entity type="TIME">1889</entity></entity> à <entity type="LOC">Paris</entity>, organisée au <entity type="LOC">Champ-de-Mars</entity>, <entity type="PERS">Gustave Eiffel</entity> a présenté <entity type="PROD">la tour Eiffel</entity>, construite par la <entity type="ORG">société SETE</entity>.
</root>


Now process the following input:
{text}
"""


FEWSHOT_ADOPTED = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are fully contained within others — do not overlap separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single <root> element.
- Return ONLY valid XML — no extra comments, headers, or explanation.

Here are some examples:

[Prose]
Input:
"J'ai vu Émile Zola pour la première fois à Paris en 1888."
Output:
<root>
J'ai vu <entity type="PERS">Émile Zola</entity> pour la première fois à <entity type="LOC">Paris</entity> <entity type="TIME">en 1888</entity>.
</root>

[Poetry]
Input:
"Sous le pont Mirabeau coule la Seine."
Output:
<root>
Sous le <entity type="LOC">pont Mirabeau</entity> coule la <entity type="LOC">Seine</entity>.
</root>

[Encyclopedia]
Input:
"La Tour Eiffel est un monument de Paris, conçu par Gustave Eiffel et inauguré lors de l'Exposition universelle de 1889."
Output:
<root>
<entity type="PROD">La Tour Eiffel</entity> est un monument de <entity type="LOC">Paris</entity>, conçu par <entity type="PERS">Gustave Eiffel</entity> et inauguré lors de l'<entity type="EVENT">Exposition universelle de <entity type="TIME">1889</entity></entity>.
</root>

[Information]
Input:
"Le vaccin Comirnaty, développé par Pfizer et BioNTech, a été approuvé en décembre 2020."
Output:
<root>
Le vaccin <entity type="PROD">Comirnaty</entity>, développé par <entity type="ORG">Pfizer</entity> et <entity type="ORG">BioNTech</entity>, a été approuvé <entity type="TIME">en décembre 2020</entity>.
</root>

[Multi]
Input:
"Barack Obama a visité Berlin, puis a donné une conférence à l'université de Chicago."
Output:
<root>
<entity type="PERS">Barack Obama</entity> a visité <entity type="LOC">Berlin</entity>, puis a donné une conférence à l'<entity type="ORG">université de Chicago</entity>.
</root>

[Spoken]
Input:
"Alors euh Macron il est parti à Bruxelles pour le sommet de l’OTAN."
Output:
<root>
Alors euh <entity type="PERS">Macron</entity> il est parti à <entity type="LOC">Bruxelles</entity> pour le <entity type="EVENT">sommet de l’OTAN</entity>.
</root>

Now process the following input:
{text}
"""


COT_ZEROSHOT = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.


Reasoning step (do not include this in the final output):
1. Carefully read the input text and detect spans that look like named entities.
2. Decide for each candidate which category (PERS, LOC, ORG, TIME, PROD, EVENT) it belongs to.
3. Make sure to respect the rules: use only these six categories, annotate every occurrence, and handle nesting only when fully contained.
4. Once the reasoning is complete, produce the final answer in strict XML format.

Now process the following input. Think step by step, but at the very end return ONLY the valid XML:
{text}
"""
# 1 EXAMPLE HELPING LLM UNDERTANDING THE FORMAT
COT_ONESHOT = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.
- Return ONLY valid XML — no extra comments, headers, or explanation.


Here is an example:
Input:
"Paris est la capitale de la France."

Output:
<root>
<entity type="LOC">Paris</entity> est la capitale de la <entity type="LOC">France</entity>.
</root>

Reasoning step (do not include this in the final output):
1. Carefully read the input text and detect spans that look like named entities.
2. Decide for each candidate which category (PERS, LOC, ORG, TIME, PROD, EVENT) it belongs to.
3. Make sure to respect the rules: use only these six categories, annotate every occurrence, and handle nesting only when fully contained.
4. Once the reasoning is complete, produce the final answer in strict XML format.

Now process the following input. Think step by step, but at the very end return ONLY the valid XML:
{text}
"""
#few shot, including nesteld entities example
COT_FEWSHOT = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are **fully contained** within others — do **not overlap** separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single `<root>` element.
- Return ONLY valid XML — no extra comments, headers, or explanation.


Here are some examples:
Input:
"Pendant l'exposition universelle de 1889 à Paris, organisée au Champ-de-Mars, Gustave Eiffel a présenté la tour Eiffel, construite par la société SETE."

Output:
<root>
Pendant l'<entity type="EVENT">exposition universelle de <entity type="TIME">1889</entity></entity> à <entity type="LOC">Paris</entity>, organisée au <entity type="LOC">Champ-de-Mars</entity>, <entity type="PERS">Gustave Eiffel</entity> a présenté <entity type="PROD">la tour Eiffel</entity>, construite par la <entity type="ORG">société SETE</entity>.
</root>

Input:
"L’ONU a été fondée en 1945 à San Francisco."

Output:
<root>
L’<entity type="ORG">ONU</entity> a été fondée <entity type="TIME">en 1945</entity> à <entity type="LOC">San Francisco</entity>.
</root>

Reasoning step (do not include this in the final output):
1. Carefully read the input text and detect spans that look like named entities.
2. Decide for each candidate which category (PERS, LOC, ORG, TIME, PROD, EVENT) it belongs to.
3. Make sure to respect the rules: use only these six categories, annotate every occurrence, and handle nesting only when fully contained.
4. Once the reasoning is complete, produce the final answer in strict XML format.

Now process the following input. Think step by step, but at the very end return ONLY the valid XML:
{text}
"""

COT_FEWSHOT_ADOPTED = """You are a high-precision named entity recognizer.

Your task is to annotate the following French text using XML-style inline tags. Your output must follow these rules exactly:

Use this format to annotate named entities:
<entity type="TYPE">ENTITY TEXT</entity>

Where TYPE is one of the following six categories:
- PERS: Person names or named groups of people.
- LOC: Places such as cities, countries, landmarks.
- ORG: Named organizations (companies, institutions, associations).
- TIME: Temporal expressions (dates, hours, named time periods).
- PROD: Named products (books, artworks, software, etc.).
- EVENT: Named events (festivals, wars, conferences, etc.).

Output Requirements:
- Wrap each entity **in the original input** using the above XML tag.
- **Do not remove, alter, or add any content** beyond inserting the tags.
- If the same entity appears more than once, annotate each occurrence.
- Use **nested tags** only when entities are fully contained within others — do not overlap separate entities.
- If no entities are present, return the input text unchanged.
- Wrap the entire output in a single <root> element.
- Return ONLY valid XML — no extra comments, headers, or explanation.

Here are some examples:

[Prose]
Input:
"J'ai vu Émile Zola pour la première fois à Paris en 1888."
Output:
<root>
J'ai vu <entity type="PERS">Émile Zola</entity> pour la première fois à <entity type="LOC">Paris</entity> <entity type="TIME">en 1888</entity>.
</root>

[Poetry]
Input:
"Sous le pont Mirabeau coule la Seine."
Output:
<root>
Sous le <entity type="LOC">pont Mirabeau</entity> coule la <entity type="LOC">Seine</entity>.
</root>

[Encyclopedia]
Input:
"La Tour Eiffel est un monument de Paris, conçu par Gustave Eiffel et inauguré lors de l'Exposition universelle de 1889."
Output:
<root>
<entity type="PROD">La Tour Eiffel</entity> est un monument de <entity type="LOC">Paris</entity>, conçu par <entity type="PERS">Gustave Eiffel</entity> et inauguré lors de l'<entity type="EVENT">Exposition universelle de <entity type="TIME">1889</entity></entity>.
</root>

[Information]
Input:
"Le vaccin Comirnaty, développé par Pfizer et BioNTech, a été approuvé en décembre 2020."
Output:
<root>
Le vaccin <entity type="PROD">Comirnaty</entity>, développé par <entity type="ORG">Pfizer</entity> et <entity type="ORG">BioNTech</entity>, a été approuvé <entity type="TIME">en décembre 2020</entity>.
</root>

[Multi]
Input:
"Barack Obama a visité Berlin, puis a donné une conférence à l'université de Chicago."
Output:
<root>
<entity type="PERS">Barack Obama</entity> a visité <entity type="LOC">Berlin</entity>, puis a donné une conférence à l'<entity type="ORG">université de Chicago</entity>.
</root>

[Spoken]
Input:
"Alors euh Macron il est parti à Bruxelles pour le sommet de l’OTAN."
Output:
<root>
Alors euh <entity type="PERS">Macron</entity> il est parti à <entity type="LOC">Bruxelles</entity> pour le <entity type="EVENT">sommet de l’OTAN</entity>.
</root>


Reasoning step (do not include this in the final output):
1. Carefully read the input text and detect spans that look like named entities.
2. Decide for each candidate which category (PERS, LOC, ORG, TIME, PROD, EVENT) it belongs to.
3. Make sure to respect the rules: use only these six categories, annotate every occurrence, and handle nesting only when fully contained.
4. Once the reasoning is complete, produce the final answer in strict XML format.

Now process the following input. Think step by step, but at the very end return ONLY the valid XML:
{text}
"""


SYSTEM_PROMPT_STYLE = {
    "simple": SIMPLE_PROMPT_DEFAULT, #with no definitions
    "zero-shot": ZEROSHOT_PROMPT, #with definitions
    "complex": SIMPLE_PROMPT_4SHOTS, #with 4 examples

    "cot_simple": COT_SIMPLE_PROMPT, #cot version of SYSTEM_PROMPT_DEFAULT
    "cot_zero_shot": COT_ZEROSHOT_PROMPT, #cot version of zero shot, SIMPLE_PROMPT,  break prompt into clearly labeled steps, with numbered instructions.
    "cot_withexamples": COT_PROMPT_4SHOTS,#rephrased cot version of complex, with the same examples as complex, 4 examples

    "zeroshot_balise": ZEROSHOT_PROMPT_BALISE,#zeroshot with balise
    "oneshot_balise": ONESHOT_PROMPT_BALISE, # 1 example helping LLM understanding the format, with balise
    "fewshot_balise": FEWSHOT_PROMPT_BALISE, #fewshot with balise, with nested entities example

    "fewshot_adopted": FEWSHOT_ADOPTED, #adopted fewshot, one example phrase per genre, 6 shots in general

    "cot_zeroshot" : COT_ZEROSHOT,
    "cot_oneshot" : COT_ONESHOT,
    "cot_fewshot" : COT_FEWSHOT,
    "cot_fewshot_adopted" : COT_FEWSHOT_ADOPTED
} 



