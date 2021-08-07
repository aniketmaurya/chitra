from unittest.mock import patch

import typer
from typer.testing import CliRunner

from chitra.cli import builder

runner = CliRunner()


@patch("chitra.cli.builder.os")
@patch("chitra.cli.builder.file_check")
def test_app(mock_file_check, mock_os):
    app = typer.Typer()
    app.command()(builder.create)
    result = runner.invoke(app, input="N\n")
    assert result.exit_code == 0
