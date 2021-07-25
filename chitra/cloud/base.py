class Cloud:
    PROVIDER = None

    def upload_artifacts(self, target_path: str):
        raise NotImplementedError

    def dockerize(self):
        raise NotImplementedError
