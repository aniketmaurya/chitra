from chitra.serve import create_api

model = lambda x: x
app = create_api(model, run=True, api_type='question-ans')
