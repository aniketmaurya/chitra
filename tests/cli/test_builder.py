from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from chitra.cli import builder

runner = CliRunner()


@patch("chitra.cli.builder.subprocess")
@patch("chitra.cli.builder.file_check")
def test_app(mock_file_check, mock_subprocess):
    mock_file_check.return_value = True
    mock_subprocess.run = MagicMock()
    app = typer.Typer()
    app.command()(builder.create)
    result = runner.invoke(app, input="N\n")
    assert result.exit_code == 0
