from ..processor import DataProcessor


class DefaultTextProcessor:
    nlp = DataProcessor(lambda x: x, lambda x: x)
