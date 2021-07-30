import typer
from typer.testing import CliRunner

from chitra.cli.dockerize import run

runner = CliRunner()


def test_app():
    app = typer.Typer()
    app.command()(run)
    result = runner.invoke(app, input="N\n")
    assert result.exit_code == 0
