from chitra.serve import create_api


def model(x):
    return x


app = create_api(model, run=True, api_type="question-ans")
