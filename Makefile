.PHONY: test test-gateway test-model train up down smoke load

test: test-gateway test-model

test-gateway:
	cd gateway && go test -race ./...

test-model:
	cd model && python -m pytest -q

train:
	cd model && python -m app.train

up:
	docker compose up --build -d

down:
	docker compose down

smoke:
	./scripts/smoke.sh

load:
	cd gateway && go run ./cmd/loadtest -n 2000 -c 50
