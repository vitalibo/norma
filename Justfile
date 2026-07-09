# show available recipes
default:
    @just --list

# update lockfile
lock:
    uv lock

# install dependencies
sync:
    uv sync --locked --dev

# run code formatter
format *args:
    uv run isort ./src/ ./tests/ {{ args }}

# run code style checks
check *args:
    uv run ruff check --config ruff.toml {{ args }} ./src/ ./tests/

# run unit tests
test *args:
    uv run -m pytest -v -p no:cacheprovider --disable-warnings {{ args }} ./tests/

# run unit tests with coverage
coverage *args:
    just test --cov=norma --cov-branch {{ args }}

# build package
build *args:
    uv build --sdist --wheel {{ args }}

# clean workdir
clean:
    rm -rf ./.pytest_cache ./build ./dist ./src/*.egg-info ./.coverage
