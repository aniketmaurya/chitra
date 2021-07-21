from chitra.serve.data_processing import DataProcessor


class DefaultTextProcessor:
    nlp = DataProcessor(lambda x: x, lambda x: x)
