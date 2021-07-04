# from chitra.serve import create_api

# app = create_api(lambda x: x, run=True, api_type='question-ans')

import numpy as np

from chitra.serve.model_serverv2 import API


def model(x):
    return np.asarray([1])


api = API(api_type="image-classification", model=model)

api.run()
