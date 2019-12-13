.PHONY: build
build:
	docker build -t registry.nathanmartins.codes/son .

.PHONY: publish
publish:
	docker push registry.nathanmartins.codes/son

.PHONY: tests
tests:
	docker run -it --rm --name son registry.nathanmartins.codes/son:latest python -m unittest discover -v -f  tests

# Run specific tests by calling like such:
# make test_name=tests.random_test unit-test
.PHONY: unit-test
unit-test:
	docker run -it --rm --name son registry.nathanmartins.codes/son:latest python  -m unittest -v $(test_name)

.PHONY: ssh
ssh:
	docker run -it --rm --name son registry.nathanmartins.codes/son:latest /bin/bash

.PHONY: run
run:
	docker run -it --rm --name son registry.nathanmartins.codes/son:latest
