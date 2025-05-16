
install-dependencies:
	@echo "--> Installing Python dependencies"
	@pip --disable-pip-version-check install -r requirements.txt
	@echo ""

lint-check:
	@echo "--> Running linter check"
	autopep8 --in-place --recursive src
	flake8 **/*.py
	black --check src
	pylama src
	isort -c src
	pylint $(git ls-files '*.py')

lint:
	@echo "--> Running linter"
	black src
	flake8 src
	isort src

dvc:
	@dvc pull

test:
	@echo "--> Running Test"
	@poetry run  pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t  hmc-torch:0.0.1  .
