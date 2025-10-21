import json
import os

def clean_chat_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned_messages = []
    skipped_messages = 0

    for message in data['messages']:
        if message['type'] != 'message':
            skipped_messages += 1
            continue

        text_content = ""
        if isinstance(message['text'], str):
            text_content = message['text']
        elif isinstance(message['text'], list):
            for entity in message['text']:
                if isinstance(entity, dict) and 'text' in entity:
                    text_content += str(entity['text'])
                elif isinstance(entity, str):
                    text_content += entity

        if not text_content.strip():
            skipped_messages += 1
            continue

        cleaned_messages.append({
            'author': message['from'],
            'text': text_content.strip()
        })

    return cleaned_messages

if __name__ == "__main__":
    input_file_path = "your_file.json"
    base_name = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_name}_cleaned.json"
    cleaned_messages = clean_chat_data(input_file_path)
    output_data = {
        "messages": cleaned_messages
    }

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)