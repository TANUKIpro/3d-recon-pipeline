.PHONY: build up down

# Write current git branch to .git_branch (read by the dashboard at runtime)
.git_branch:
	@git rev-parse --abbrev-ref HEAD > .git_branch
	@echo "Branch: $$(cat .git_branch)"

build: .git_branch
	docker compose build

up: .git_branch
	docker compose up

down:
	docker compose down
