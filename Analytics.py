import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
def load_emotion_model():
    model_name = "cointegrated/rubert-tiny2-cedr-emotion-detection"
    emotion_classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True
    )
    return emotion_classifier
emotion_mapping = {
    'neutral': 'нейтральное',
    'sadness': 'грусть',
    'fear': 'страх',
    'anger': 'злость',
    'joy': 'радость'
}

def analyze_emotion(text, classifier):
    if len(text.strip()) == 0:
        return 'нейтральное'
    try:
        results = classifier(text)[0]
        max_emotion = max(results, key=lambda x: x['score'])
        emotion_en = max_emotion['label']
        return emotion_mapping.get(emotion_en, 'нейтральное')
    except Exception as e:
        print(f"Ошибка при анализе текста: {e}")
        return 'нейтральное'

def analyze_chat_emotions(input_file, output_file):
    emotion_classifier = load_emotion_model()
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    analyzed_messages = []

    for i, message in enumerate(data['messages']):
        if i % 100 == 0:
            print(f"Обработано {i}/{len(data['messages'])} сообщений...")

        emotion = analyze_emotion(message['text'], emotion_classifier)

        analyzed_messages.append({
            'author': message['author'],
            'emotion': emotion,
            'text': message['text'][:100] + "..." if len(message['text']) > 100 else message['text']
        })
    output_data = {
        "analysis_info": {
            "model": "cointegrated/rubert-tiny2-cedr-emotion-detection",
            "total_messages": len(analyzed_messages),
            "emotion_mapping": emotion_mapping
        },
        "messages": analyzed_messages
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    emotion_stats = {}
    for msg in analyzed_messages:
        emotion = msg['emotion']
        emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

    print("\nСтатистика эмоций:")
    for emotion, count in emotion_stats.items():
        percentage = (count / len(analyzed_messages)) * 100
        print(f"  {emotion}: {count} сообщений ({percentage:.1f}%)")
if __name__ == "__main__":
    import os
    import glob
    cleaned_files = glob.glob("*_cleaned.json")

    if cleaned_files:
        input_file = cleaned_files[0]
        output_file = input_file.replace("_cleaned.json", "_emotions.json")
    else:
        input_file = "your_file_cleaned.json"
        output_file = "chat_emotions.json"
    analyze_chat_emotions(input_file, output_file)