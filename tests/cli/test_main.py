import typer
from typer.testing import CliRunner

from chitra.cli.main import version

runner = CliRunner()


def test_app():
    app = typer.Typer()
    app.command()(version)
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "You're running chitra version" in result.stdout
