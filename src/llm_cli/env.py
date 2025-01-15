import dataclasses
import pathlib

import decouple


env_file = pathlib.Path(__file__).parents[2] / ".env"
env_config = decouple.Config(decouple.RepositoryEnv(env_file))


@dataclasses.dataclass(frozen=True)
class EnvironmentVariableNotSet(Exception):
    key: str


def as_str(key: str) -> str:
    try:
        return env_config.get(key)
    except decouple.UndefinedValueError as exc:
        raise EnvironmentVariableNotSet(key=key) from exc
