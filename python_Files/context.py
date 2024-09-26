from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentence = "let's go grab dinner in a 5-star hotel" # Input sentence

# labels to classify the sentence
labels = ["beach vacation", "mountain vacation", "city trip", "island getaway", "cultural experience", "adventure trip", 
          "cruise vacation", "safari trip", "historical tour", "nature retreat", "skiing trip", "desert adventure", 
          "wellness retreat", "road trip", "wildlife tour", "festive tour", "hiking trip", "rural village experience", 
          "national park visit", "countryside escape", "spiritual retreat", "wine tasting tour", "luxury resort vacation", 
          "camping tour", "backpacking adventure", "amusement park visit", "shopping spree", "fishing trip", "photography tour", 
          "foodie tour"] 

classification = classifier(sentence, candidate_labels=labels)
predicted_label = classification['labels'][0]  
confidence_score = classification['scores'][0] 
print(f"Predicted label: {predicted_label}")
print(f"Confidence score: {confidence_score:.4f}")
