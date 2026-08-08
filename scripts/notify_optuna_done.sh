#!/bin/bash
# optuna(200/200): 완료 알림 + random+optuna 결합 요약.
# autoresearch(200/200): 완료 알림 + random+optuna+autoresearch 전체 결합 요약.
# nohup+disown으로 실행하여 SSH/Claude 세션과 무관하게 pod에서 독립 동작.
TELEGRAM_BOT_TOKEN="8904393112:AAGVxbubIV_WuqU4fV-JbuKmlg8od9XOzBA"
TELEGRAM_CHAT_ID="257404019"
TSV=/workspace/Masters_degree/results/phase3_autoresearch/results.tsv
PY=/workspace/Masters_degree/.venv/bin/python
START_TIME=$(date -u +%s)

send_msg() {
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"     -d chat_id="${TELEGRAM_CHAT_ID}"     -d text="$1" >/dev/null
}

count_stage() {
  tail -n +2 "$TSV" 2>/dev/null | awk -F'\t' -v s="$1" '$2==s' | wc -l
}

# 1) optuna 200/200 대기
while true; do
  COUNT=$(count_stage optuna)
  if [ "$COUNT" -ge 200 ]; then
    ELAPSED_H=$(( ($(date -u +%s) - START_TIME) / 3600 ))
    cd /workspace/Masters_degree
    SUMMARY=$("$PY" scripts/summarize_stage.py random optuna 2>&1)
    BEST_LINES=$(echo "$SUMMARY" | grep 'best trial:')
    send_msg "[Phase3] optuna 200/200 완료! ($(date -u '+%Y-%m-%d %H:%M UTC'), 감시 시작 후 약 ${ELAPSED_H}시간 경과)
${BEST_LINES}
전체 요약(random+optuna): results/phase3_autoresearch/phase3_summary.txt
autoresearch 진행: $(count_stage autoresearch)/200"
    break
  fi
  sleep 300
done

# 2) autoresearch 200/200 대기 (Phase3 전체 완료)
while true; do
  COUNT=$(count_stage autoresearch)
  if [ "$COUNT" -ge 200 ]; then
    ELAPSED_H=$(( ($(date -u +%s) - START_TIME) / 3600 ))
    cd /workspace/Masters_degree
    SUMMARY=$("$PY" scripts/summarize_stage.py random optuna autoresearch 2>&1)
    BEST_LINES=$(echo "$SUMMARY" | grep 'best trial:')
    send_msg "[Phase3] autoresearch 200/200 완료! Phase3 전체 종료. ($(date -u '+%Y-%m-%d %H:%M UTC'), 감시 시작 후 약 ${ELAPSED_H}시간 경과)
${BEST_LINES}
전체 요약(random+optuna+autoresearch): results/phase3_autoresearch/phase3_summary.txt"
    break
  fi
  sleep 300
done
