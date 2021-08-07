from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from chitra.cli import builder
from chitra.cli.builder import file_check

runner = CliRunner()


@patch("chitra.cli.builder.subprocess")
@patch("chitra.cli.builder.file_check")
def test_app(mock_file_check, mock_subprocess):
    mock_file_check.return_value = True
    mock_subprocess.run = MagicMock()
    mock_subprocess.run.return_value = MagicMock()
    mock_subprocess.run.return_value.returncode = True

    app = typer.Typer()
    app.command()(builder.create)
    result = runner.invoke(app, input="Y\n")
    assert result.exit_code == 0


def test_file_check():
    with pytest.raises(UserWarning):
        file_check(["requirements.txt"])

    with pytest.raises(UserWarning):
        file_check([])
