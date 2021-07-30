import typer

from chitra.cli import dockerize, examples

app = typer.Typer(name="chitra CLI ✨")

app.add_typer(dockerize.app, name="dockerizer")
app.add_typer(examples.app, name="examples")
