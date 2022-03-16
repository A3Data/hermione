from click.testing import CliRunner
from hermione.cli.main import cli, logo


def test_installation_ok():
    runner = CliRunner()
    res = runner.invoke(cli)
    assert res.exit_code == 0


def test_info():
    runner = CliRunner()
    res = runner.invoke(cli, ["info"])
    assert logo in res.output
