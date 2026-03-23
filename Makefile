.PHONY: build up down

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)
export GIT_BRANCH

build:
	@echo "Branch: $(GIT_BRANCH)"
	docker compose build

up:
	@echo "Branch: $(GIT_BRANCH)"
	docker compose up

down:
	docker compose down
