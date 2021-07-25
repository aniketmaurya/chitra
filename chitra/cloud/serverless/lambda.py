from ..base import Cloud


class LambdaDeployer(Cloud):
    PROVIDER = "aws"

    def __init__(self, user: str, password: str, region: str):
        """"""
