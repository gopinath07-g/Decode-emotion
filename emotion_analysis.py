
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load a pretrained emotion detection model
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for emotion classification
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Example social media posts
texts = [
    "I'm so happy with the new update!",
    "I feel really anxious about the future.",
    "That movie made me cry.",
    "Why is customer service so bad?",
    "What a beautiful day!"
]

# Analyze emotions
for text in texts:
    print(f"\nText: {text}")
    results = emotion_classifier(text)[0]
    for res in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{res['label']}: {res['score']:.4f}")
