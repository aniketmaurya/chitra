import typer

from chitra.cli import dockerizer, examples

app = typer.Typer(name="chitra CLI ✨")

app.add_typer(dockerizer.app, name="dockerizer")
app.add_typer(examples.app, name="examples")
