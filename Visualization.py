import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import os

def load_emotion_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def emotion_to_mood_score(emotion):
    emotion_scores = {
        'радость': 9,
        'нейтральное': 5,
        'грусть': 3,
        'злость': 2,
        'страх': 1
    }
    return emotion_scores.get(emotion, 5)

def create_mood_timeline_sequential(data, output_path):
    df = pd.DataFrame(data['messages'])
    df['message_order'] = range(1, len(df) + 1)
    df['mood_score'] = df['emotion'].apply(emotion_to_mood_score)
    window_size = max(10, len(df) // 20)
    authors = df['author'].unique()
    print(f"Найдены авторы: {list(authors)}")
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, author in enumerate(authors):
        author_data = df[df['author'] == author].copy()
        print(f"Автор '{author}': {len(author_data)} сообщений")
        author_data = author_data.sort_values('message_order')
        author_data['mood_smooth'] = author_data['mood_score'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        ax.plot(author_data['message_order'], author_data['mood_smooth'],
               color=colors[i % len(colors)], linewidth=3, label=f'{author}', alpha=0.8)
        if len(author_data) < 500:
            ax.scatter(author_data['message_order'], author_data['mood_score'],
                      alpha=0.2, color=colors[i % len(colors)], s=10)
    ax.set_title('Динамика настроения в переписке\n(по порядку сообщений)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Порядковый номер сообщения', fontsize=12)
    ax.set_ylabel('Уровень настроения', fontsize=12)
    mood_levels = ['Ужасное', 'Плохое', 'Нейтральное', 'Хорошее', 'Отличное']
    mood_positions = [1, 3, 5, 7, 9]
    ax.set_yticks(mood_positions)
    ax.set_yticklabels(mood_levels)
    ax.set_ylim(0, 10)
    ax.axhspan(0, 2, alpha=0.1, color='red', label='Отрицательное')
    ax.axhspan(2, 4, alpha=0.1, color='orange', label='Слегка отрицательное')
    ax.axhspan(4, 6, alpha=0.1, color='gray', label='Нейтральное')
    ax.axhspan(6, 8, alpha=0.1, color='lightgreen', label='Слегка положительное')
    ax.axhspan(8, 10, alpha=0.1, color='green', label='Положительное')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_path}_mood_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_emotion_pie_charts(data, output_path):
    df = pd.DataFrame(data['messages'])
    authors = df['author'].unique()
    emotion_colors = {
        'радость': '#4CAF50',
        'нейтральное': '#9E9E9E',
        'грусть': '#FF9800',
        'злость': '#F44336',
        'страх': '#795548'
    }
    if len(authors) > 1:
        fig, axes = plt.subplots(1, len(authors), figsize=(6 * len(authors), 5))
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        axes = [ax]

    for i, author in enumerate(authors):
        author_data = df[df['author'] == author]
        emotion_counts = author_data['emotion'].value_counts()
        print(f"Автор '{author}': {dict(emotion_counts)}")
        current_colors = [emotion_colors.get(emotion, '#CCCCCC') for emotion in emotion_counts.index]
        wedges, texts, autotexts = axes[i].pie(
            emotion_counts.values,
            labels=emotion_counts.index,
            colors=current_colors,
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
            startangle=90,
            textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        for text in texts:
            text.set_fontsize(9)
        axes[i].set_title(f'Эмоции: {author}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_path}_emotion_pie_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_mood_comparison_chart(data, output_path):
    df = pd.DataFrame(data['messages'])
    authors = df['author'].unique()
    print(f"Сравнительная гистограмма для авторов: {list(authors)}")
    author_scores = []
    for author in authors:
        author_data = df[df['author'] == author]
        avg_score = author_data['emotion'].apply(emotion_to_mood_score).mean()
        author_scores.append(avg_score)
        print(f"Автор '{author}': средний балл = {avg_score:.2f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(authors, author_scores, color=colors[:len(authors)], alpha=0.7)
    for bar, score in zip(bars, author_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Средний уровень настроения по авторам', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Средний балл настроения (0-10)', fontsize=12)
    ax.set_ylim(0, 10)
    for y in [2, 4, 6, 8]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    mood_labels = ['0: Ужасное', '2: Плохое', '4: Нейтральное', '6: Хорошее', '8: Отличное', '10: Идеальное']
    for i, label in enumerate(mood_labels):
        ax.text(1.02, i*2, label, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_path}_mood_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_visualizations():
    emotion_files = [f for f in os.listdir('.') if 'emotion' in f and f.endswith('.json')]
    if not emotion_files:
        print("Файл с эмоциями не найден. Сначала запустите анализ эмоций.")
        return
    emotion_file = emotion_files[0]
    data = load_emotion_data(emotion_file)
    base_name = emotion_file.replace('.json', '')
    create_mood_timeline_sequential(data, base_name)
    create_emotion_pie_charts(data, base_name)
    create_mood_comparison_chart(data, base_name)
    df = pd.DataFrame(data['messages'])

    for author in df['author'].unique():
        author_data = df[df['author'] == author]
        total_messages = len(author_data)
        mood_scores = author_data['emotion'].apply(emotion_to_mood_score)
        avg_mood = mood_scores.mean()
        emotion_stats = author_data['emotion'].value_counts()
        print(f"\n{author} (всего сообщений: {total_messages}):")
        print(f"  Средний балл настроения: {avg_mood:.2f}/10")
        for emotion, count in emotion_stats.items():
            percentage = (count / total_messages) * 100
            mood_score = emotion_to_mood_score(emotion)
            print(f"  {emotion}: {count} сообщ. ({percentage:.1f}%) [оценка: {mood_score}/10]")
    print(f"\nВизуализации сохранены в файлы:")
    print(f"  - {base_name}_mood_timeline.png")
    print(f"  - {base_name}_emotion_pie_charts.png")
    print(f"  - {base_name}_mood_comparison.png")
if __name__ == "__main__":
    create_visualizations()