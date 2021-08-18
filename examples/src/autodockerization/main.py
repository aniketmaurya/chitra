from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from chitra.serve import create_api

tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/xtremedistil-l6-h256-uncased"
)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

api = create_api(classifier, run=False, api_type="text-classification")

app = api.app
