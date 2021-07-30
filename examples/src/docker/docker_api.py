from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from chitra.docker.api_docker import dockerize_api
from chitra.serve import create_api

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/beto-sentiment-analysis"
)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

api = create_api(classifier, run=False, api_type="text-classification")


docker_cmd = dockerize_api("./", main_module_name='docker_api', )
print(docker_cmd)
