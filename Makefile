.PHONY: help setup deps-upstreams deps-autoresearch deps-cevolve format lint typecheck pylint test test-unit test-integration test-full test-upstream-live test-live test-max-live test-all build check doctor strict-env-status strict-env-validate

.DEFAULT_GOAL := help

help:
	@echo "autoclanker Make Targets"
	@echo "  make setup"
	@echo "  make deps-upstreams"
	@echo "  make deps-autoresearch"
	@echo "  make deps-cevolve"
	@echo "  make format"
	@echo "  make lint"
	@echo "  make typecheck"
	@echo "  make pylint"
	@echo "  make test"
	@echo "  make test-unit"
	@echo "  make test-integration"
	@echo "  make test-full"
	@echo "  make test-upstream-live"
	@echo "  make test-live"
	@echo "  make test-max-live"
	@echo "  make test-all"
	@echo "  make build"
	@echo "  make check"
	@echo "  make doctor"
	@echo "  make strict-env-status"
	@echo "  make strict-env-validate"

setup:
	./bin/dev setup

deps-upstreams:
	./bin/dev deps upstreams

deps-autoresearch:
	./bin/dev deps autoresearch

deps-cevolve:
	./bin/dev deps cevolve

format:
	./bin/dev format

lint:
	./bin/dev lint

typecheck:
	./bin/dev typecheck

pylint:
	./bin/dev pylint

test:
	./bin/dev test

test-unit:
	./bin/dev test-unit

test-integration:
	./bin/dev test-integration

test-full:
	./bin/dev test-full

test-upstream-live:
	./bin/dev test-upstream-live

test-live:
	./bin/dev test-live

test-max-live:
	./bin/dev test-max-live

test-all:
	./bin/dev test-all

build:
	./bin/dev build

check:
	./bin/dev check

doctor:
	./bin/dev doctor

strict-env-status:
	./bin/dev strict-env status

strict-env-validate:
	./bin/dev strict-env validate
