install:
	pip3 install -r requirements-dev.txt

style:
	isort ./src/ ./tests/ -l 120 -m 3
	pylint ./src/ ./tests/ --rcfile=.pylintrc

test:
	PYTHONPATH='./src' pytest -v -p no:cacheprovider --disable-warnings ./tests/

build: clean
	python3 -m build

clean:
	rm -rf ./.pytest_cache ./build ./dist ./src/*.egg-info

.PHONY: install style test build clean
