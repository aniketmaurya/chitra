from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from chitra.serve import create_api

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/beto-sentiment-analysis"
)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

api = create_api(classifier, run=False, api_type="text-classification")

app = api.app
