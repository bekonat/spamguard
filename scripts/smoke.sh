#!/usr/bin/env bash
set -euo pipefail

URL=${URL:-http://localhost:8080}

for _ in $(seq 1 60); do
  curl -fs "$URL/readyz" >/dev/null && break
  sleep 1
done
curl -fs "$URL/readyz"

check() {
  local text=$1 want=$2
  local got
  got=$(curl -fs -X POST "$URL/api/v1/classify" -H 'Content-Type: application/json' \
    -d "{\"text\":\"$text\"}" | grep -o '"label":"[a-z]*"')
  if [[ $got != "\"label\":\"$want\"" ]]; then
    echo "FAIL: '$text' -> $got, want $want" >&2
    exit 1
  fi
  echo "ok: $want <- $text"
}

check "Вы выиграли iPhone! Заберите приз по ссылке http://win-bonus.top/x1" spam
check "Привет, лекция по алгоритмам завтра в 10:00" ham

curl -fs -X POST "$URL/api/v1/classify/batch" -H 'Content-Type: application/json' \
  -d '{"texts":["Срочный займ без проверок!","Купи хлеб по дороге домой"]}'
echo
curl -fs "$URL/metrics" | grep -E '^gateway_(cache_lookups|model_batch_size_count)'
