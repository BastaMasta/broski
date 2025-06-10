medical_corpus = [
    "The patient was prescribed antibiotics to treat the bacterial infection.",
    "Diabetes mellitus is characterized by high blood sugar levels.",
    "MRI and CT scans are important tools for diagnosing brain and spinal cord injuries.",
    "Cardiovascular diseases include conditions like heart attack, stroke, and hypertension.",
    "Physical therapy helps patients recover mobility after surgery or injury.",
    "Vaccinations are critical in preventing infectious diseases such as measles and influenza.",
    "Common symptoms of flu include fever, cough, sore throat, and body aches.",
    "Blood pressure monitoring is vital for patients with hypertension or cardiovascular risks.",
    "Surgical removal of tumors requires precise planning and expert care.",
    "Medication dosages must be strictly followed to avoid adverse effects.",
    "Chronic kidney disease often requires dialysis or transplantation.",
    "Asthma is a respiratory condition causing difficulty in breathing due to airway inflammation.",
    "The immune system protects the body against pathogens and foreign substances.",
    "Radiology departments use X-rays, ultrasounds, and MRIs for diagnostic imaging.",
    "Neurological disorders affect the brain, spinal cord, and nerves.",
    "Antiviral drugs are used to treat infections caused by viruses such as HIV and hepatitis.",
    "The doctor ordered blood tests to check for anemia and infection markers.",
    "Diuretics help reduce fluid buildup in patients with heart failure or kidney disease.",
    "Cholesterol levels impact the risk of developing atherosclerosis and heart disease.",
    "Emergency medical services provide urgent care for trauma and critical conditions.",
]

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess(sentence):
    return [word for word in simple_preprocess(sentence) if word not in stop_words]

tokenized_corpus = [preprocess(sentence) for sentence in medical_corpus]

from gensim.models import Word2Vec

model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1,
workers=2, epochs=100)
print("Most similar to 'disease':")
print(model.wv.most_similar('diseases', topn=5))
print("\nMost similar to 'blood':")
print(model.wv.most_similar('blood', topn=5))