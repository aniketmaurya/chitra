from chitra.serve import create_api

app = create_api(lambda x: x, run=True, api_type='question-ans')
