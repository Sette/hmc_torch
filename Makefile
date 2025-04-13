
install-dependencies:
	@echo "--> Installing Python dependencies"
	@pip --disable-pip-version-check install -r requirements.txt
	@echo ""

lint-check:
	@echo "--> Running linter check"
	flake8 **/*.py
	black --check .
	pylama hmc
	isort -c .

lint:
	@echo "--> Running linter"
	black .
	flake8 .
	isort .

dvc:
	@dvc pull

test:
	@echo "--> Running Test"
	@poetry run  pytest --verbose --cov-report term-missing --cov-report xml --cov-report html --cov=. .
	@echo ""

build:
	@echo "--> Docker build"
	docker build -f Dockerfile -t  hmc-torch:0.0.1  .
