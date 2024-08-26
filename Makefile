
install-dependencies:
	@echo "--> Installing Python dependencies"
	@pip --disable-pip-version-check install -r requirements.txt
	@echo ""

lint-check:
	@echo "--> Running linter check"
	black --check .
	flake8 **/*.py
	pylama app
	isort -c .

lint:
	@echo "--> Running linter"
	black .
	flake8 .
	pylama app
	isort .

dvc:
	@dvc pull

test:
	@echo "--> Running Test"
	@poetry run  pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t  fma-prep:0.0.1  .
