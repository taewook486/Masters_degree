#!/bin/bash
# optuna(200/200): 완료 알림 + random+optuna 결합 요약.
# autoresearch(200/200): 완료 알림 + random+optuna+autoresearch 전체 결합 요약.
# nohup+disown으로 실행하여 SSH/Claude 세션과 무관하게 pod에서 독립 동작.
REPO=/workspace/Masters_degree

# 토큰은 코드에 두지 않는다. REPO/.env(gitignore 대상)나 셸 환경에서 읽는다.
# 과거 이 파일에 토큰을 하드코딩해 공개 저장소로 유출된 적 있음(커밋 a715c3f).
[ -f "$REPO/.env" ] && . "$REPO/.env"

if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
  echo "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 가 없습니다. $REPO/.env 에 설정하세요." >&2
  exit 1
fi

TSV=$REPO/results/phase3_autoresearch/results.tsv
PY=$REPO/.venv/bin/python
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
    cd "$REPO"
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
    cd "$REPO"
    SUMMARY=$("$PY" scripts/summarize_stage.py random optuna autoresearch 2>&1)
    BEST_LINES=$(echo "$SUMMARY" | grep 'best trial:')
    send_msg "[Phase3] autoresearch 200/200 완료! Phase3 전체 종료. ($(date -u '+%Y-%m-%d %H:%M UTC'), 감시 시작 후 약 ${ELAPSED_H}시간 경과)
${BEST_LINES}
전체 요약(random+optuna+autoresearch): results/phase3_autoresearch/phase3_summary.txt

■ 지금 바로 RunPod 콘솔에서 팟을 Stop 하세요. 켜둔 만큼 계속 과금됩니다.
  https://www.runpod.io/console/pods"
    break
  fi
  sleep 300
done

# 3) 팟이 꺼질 때까지 1시간마다 재알림.
#    실험이 끝난 뒤 pod를 방치해 하루 반치 요금이 나간 적 있음(2026-08-13).
#    pod가 Stop되면 이 스크립트도 함께 죽으므로 알림은 자동으로 멈춘다.
IDLE_START=$(date -u +%s)
while true; do
  sleep 3600
  IDLE_H=$(( ($(date -u +%s) - IDLE_START) / 3600 ))
  send_msg "[Phase3] 실험은 끝났는데 팟이 아직 켜져 있습니다 (완료 후 ${IDLE_H}시간 경과).
지금 Stop 하지 않으면 요금이 계속 나갑니다: https://www.runpod.io/console/pods"
done
