.PHONY: setup data run figures report test lint all

setup:
	uv sync --locked

data:
	uv run otpf data

run: data
	uv run otpf run
	uv run otpf figures

figures:
	uv run otpf figures

report:
	uv run otpf report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: setup run report
