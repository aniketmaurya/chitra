from unittest.mock import patch

import typer
from typer.testing import CliRunner

from chitra.cli import builder

runner = CliRunner()


@patch("chitra.cli.dockerize.os")
@patch("chitra.cli.dockerize.file_check")
def test_app(mock_file_check, mock_os):
    app = typer.Typer()
    app.command()(builder.run)
    result = runner.invoke(app, input="N\n")
    assert result.exit_code == 0
