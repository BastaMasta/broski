from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

reviews = [
    "The product quality is amazing!",
    "I received the item.",
    "Customer service was terrible.",
    "I'm extremely satisfied with the delivery speed.",
    "The item broke after two days, very disappointed.",
    "Great value for money. Will buy again!"
]

for review in reviews:
    result = sentiment_pipeline(review)[0] # returns a list with one dict
    print(f"Review: \"{review}\"\n → Sentiment: {result['label']}, Score: {result['score']:.4f}\n")