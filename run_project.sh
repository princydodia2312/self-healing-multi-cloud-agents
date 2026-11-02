#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_project.sh — run the whole project with ONE command (Ubuntu / WSL).
#
#   ./run_project.sh            interactive menu
#   ./run_project.sh demo       start AWS + show all results (no new agent run) ← viva / submission
#   ./run_project.sh full       start AWS + seed data + run BOTH pipelines + show results
#   ./run_project.sh eu_sales   run only the PostgreSQL pipeline   (add --fresh to rewrite today's data)
#   ./run_project.sh shop_orders run only the MongoDB pipeline     (add --fresh to rewrite today's data)
#   ./run_project.sh results    show S3 files, Trino query results, Grafana URL
#   ./run_project.sh ui         Streamlit UI with a LIVE Trino connection (Ctrl+C to stop)
#   ./run_project.sh check      offline checks: tests + self-heal eval + security gate (free)
#   ./run_project.sh status     AWS power state
#   ./run_project.sh pause      stop the database to save money
#   ./run_project.sh destroy    DELETE all AWS resources (asks for confirmation)
#
# Needs a filled-in .env (see .env.example) and the one-time setup in README §6.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

cd "$(dirname "$0")" || exit 1
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"   # uv on PATH in non-login shells

REGION="eu-central-1"
CLUSTER="multi-cloud-agent-cluster"
EU_BUCKET="eu-sales-insights-data-princy2312"
SHOP_BUCKET="princy2312-shop-orders-insights-data"
LOG_DIR="logs"
TRINO_LOCAL_PORT=8080

green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red() { printf "\033[31m%s\033[0m\n" "$*"; }
step() { printf "\n\033[1;36m━━ %s ━━\033[0m\n" "$*"; }
die() { red "✖ $*"; exit 1; }

# ── Pre-flight ───────────────────────────────────────────────────────────────
load_env() {
  [ -f .env ] || die ".env not found — copy .env.example to .env and fill in the keys."
  set -a; . ./.env; set +a
  local need=(PINECONE_API_KEY GITHUB_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY) missing=()
  if [ "${LLM_PROVIDER:-gemini}" = "openai" ]; then need+=(OPENAI_API_KEY); else need+=(GEMINI_API_KEY); fi
  for k in "${need[@]}"; do [ -n "${!k:-}" ] || missing+=("$k"); done
  [ ${#missing[@]} -eq 0 ] || die "Missing in .env: ${missing[*]}"
  for t in uv aws kubectl make; do command -v "$t" >/dev/null || die "'$t' is not installed (README §6)."; done
}

check_aws() {
  aws sts get-caller-identity --query Arn --output text >/dev/null 2>&1 \
    || die "AWS login failed — run 'aws configure' or check the AWS keys in .env."
  aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER" >/dev/null 2>&1 \
    || die "EKS cluster '$CLUSTER' not found — run 'make bootstrap-aws' first (README §6)."
  green "✔ AWS login OK, kubectl connected to $CLUSTER"
}

rds_state() {
  aws rds describe-db-instances --db-instance-identifier eu-sales-raw-data \
    --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null || echo "not-found"
}

start_aws() {
  step "Starting AWS"
  check_aws
  local s; s=$(rds_state)
  if [ "$s" = "stopped" ]; then
    make -s aws-resume
  fi
  if [ "$s" != "available" ]; then
    echo "Waiting for the database (usually 5–10 min)…"
    for _ in $(seq 1 40); do
      s=$(rds_state); [ "$s" = "available" ] && break
      printf "  RDS: %s\r" "$s"; sleep 20
    done
  fi
  [ "$(rds_state)" = "available" ] || die "Database did not become available (status: $(rds_state))."
  green "✔ RDS PostgreSQL available"
  local host; host=$(aws ssm get-parameter --name /multi-cloud-self-healing-agent/aws/rds_host \
    --query Parameter.Value --output text 2>/dev/null)
  if ! timeout 10 bash -c "echo > /dev/tcp/$host/5432" 2>/dev/null; then
    yellow "⚠ The database is not reachable from this laptop (your internet IP is probably outside the"
    yellow "  allowed range). Fix: update TF_VAR_rds_allowed_cidrs in .env, then:"
    yellow "  terraform -chdir=bootstrap/aws apply"
    return 1
  fi
  green "✔ Database reachable from this laptop"
}

# ── Pipelines ────────────────────────────────────────────────────────────────
run_pipeline() {   # $1 = eu_sales | shop_orders, $2 = --fresh (optional)
  local p="$1" fresh="${2:-}" bucket
  [ "$p" = "eu_sales" ] && bucket="$EU_BUCKET" || bucket="$SHOP_BUCKET"
  step "Pipeline: $p"
  if [ "$p" = "shop_orders" ]; then
    [ -n "${MONGO_URI:-}" ] || die "MONGO_URI is missing in .env (needed for shop_orders)."
    make -s mongo-ssm
    make -s chaos target=shop_orders || die "Seeding MongoDB failed."
  else
    make -s chaos || die "Seeding PostgreSQL failed."
  fi
  if [ "$fresh" = "--fresh" ]; then
    aws s3 rm "s3://$bucket/processed/run_date=$(date +%F)/" --recursive --quiet
    green "✔ Cleared today's partition so the pipeline writes fresh data"
  elif aws s3 ls "s3://$bucket/processed/run_date=$(date +%F)/" >/dev/null 2>&1; then
    yellow "ℹ Today's partition already exists → the job will skip writing (idempotency)."
    yellow "  Use '$0 $p --fresh' to rewrite today's data."
  fi
  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/run_${p}_$(date +%Y%m%d-%H%M).log"
  echo "Agents are working (15–40 min). Full log: $log"
  echo "Watch the deploy at: https://github.com/${GITHUB_REPOSITORY:-}/actions"
  PYTHONUNBUFFERED=1 make -s run p="$p" 2>&1 | tee "$log" \
    | grep --line-buffered -E "Routing to|GATE|MISSION|CRITICAL|AUTO-VALIDATION|Medic requested|escalat" \
    | grep --line-buffered -vE "pinecone|httpx"
  if grep -q "MISSION VERIFIED" "$log"; then
    green "✅ $p: MISSION VERIFIED"
  else
    red "❌ $p did not verify — see the last lines of $log"; tail -n 15 "$log"; return 1
  fi
}

# ── Results ──────────────────────────────────────────────────────────────────
start_trino_forward() {
  pkill -f "port-forward -n analytics svc/trino" 2>/dev/null
  kubectl port-forward -n analytics svc/trino "$TRINO_LOCAL_PORT":8080 >/dev/null 2>&1 &
  PF_PID=$!
  for _ in $(seq 1 15); do
    curl -s -o /dev/null "http://localhost:$TRINO_LOCAL_PORT/v1/info" && return 0; sleep 1
  done
  return 1
}

show_results() {
  step "Results"
  check_aws
  echo "── S3 data lake (Parquet, partitioned by run_date)"
  echo "  eu_sales     (s3://$EU_BUCKET):"
  aws s3 ls "s3://$EU_BUCKET/processed/" --recursive --human-readable | grep parquet | sed 's/^/    /' || echo "    (no files yet)"
  echo "  shop_orders  (s3://$SHOP_BUCKET):"
  aws s3 ls "s3://$SHOP_BUCKET/processed/" --recursive --human-readable | grep parquet | sed 's/^/    /' || echo "    (no files yet)"

  echo; echo "── Kubernetes"
  kubectl get pods -n analytics --no-headers 2>/dev/null | awk '{print "  analytics:  "$1"  "$3}'
  kubectl get pods -n monitoring --no-headers 2>/dev/null | awk '{print "  monitoring: "$1"  "$3}'

  echo; echo "── Trino SQL over the data lake"
  if start_trino_forward; then
    TRINO_PORT_LOCAL="$TRINO_LOCAL_PORT" uv run --quiet python - <<'EOF'
import os, trino
cur = trino.dbapi.connect(host="localhost", port=int(os.environ["TRINO_PORT_LOCAL"]),
                          user="trino", catalog="hive").cursor()
def show(title, sql):
    print(f"\n  {title}")
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        print("  " + " | ".join(cols))
        for r in cur.fetchall():
            print("  " + " | ".join(str(v) for v in r))
    except Exception as e:
        print(f"  (not available: {type(e).__name__}: {str(e)[:120]})")
show("eu_sales — revenue per currency",
     "SELECT run_date, currency, COUNT(*) AS orders, ROUND(SUM(unit_price*quantity),2) AS revenue, "
     "SUM(CASE WHEN is_suspicious THEN 1 ELSE 0 END) AS suspicious "
     "FROM hive.sales_eu.pipe_eu_sales_to_s3 GROUP BY run_date, currency ORDER BY run_date DESC, revenue DESC")
show("shop_orders — revenue per city (top 5)",
     "SELECT customer_city, COUNT(*) AS items, ROUND(SUM(price*qty),2) AS revenue "
     "FROM hive.shop_orders.pipe_shop_orders_to_s3 GROUP BY customer_city ORDER BY revenue DESC LIMIT 5")
EOF
    kill "$PF_PID" 2>/dev/null
  else
    yellow "  Trino is not reachable (is the cluster running?)"
  fi

  echo; echo "── Grafana dashboard"
  local g; g=$(kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
  if [ -n "$g" ]; then
    green "  URL:  http://$g:3000"
    echo  "  User: admin   Password: run  grep GRAFANA_ADMIN_PASSWORD .env"
  else
    yellow "  Grafana not deployed yet."
  fi
  echo; echo "── GitHub Actions: https://github.com/${GITHUB_REPOSITORY:-}/actions"
}

# ── UI ───────────────────────────────────────────────────────────────────────
run_ui() {
  step "Streamlit UI (live Trino)"
  check_aws
  if start_trino_forward; then
    green "✔ Trino forwarded to localhost:$TRINO_LOCAL_PORT"
    export TRINO_HOST=localhost TRINO_PORT="$TRINO_LOCAL_PORT"
  else
    yellow "⚠ Trino not reachable — the Trino tab will show demo data."
  fi
  trap 'kill ${PF_PID:-0} 2>/dev/null; echo; green "Stopped."' EXIT
  green "Open http://localhost:8501 in your browser (Ctrl+C here to stop)."
  uv run streamlit run streamlit_app.py --server.fileWatcherType none --server.headless true
}

# ── Checks / power ───────────────────────────────────────────────────────────
run_checks() {
  step "Offline checks (no cloud, no cost)"
  uv run pytest -q 2>&1 | tail -n 1
  uv run python -m evals.harness.runner --gate 2>/dev/null | tail -n 1
  uv run python -m policy.security_analyzer . 2>/dev/null | tail -n 1
  uv run ruff check . | tail -n 1
}

confirm_destroy() {
  red "This DELETES the EKS cluster, the database, ECR and the data buckets. It cannot be undone."
  read -r -p "Type 'destroy' to continue: " ans
  [ "$ans" = "destroy" ] || { echo "Cancelled."; exit 0; }
  for b in "$EU_BUCKET" "$SHOP_BUCKET"; do aws s3 rm "s3://$b" --recursive --quiet 2>/dev/null; done
  make destroy-aws
  yellow "Also delete: the MongoDB Atlas cluster, the AWS access key (IAM), and optionally the Pinecone index."
}

menu() {
  echo "Self-Healing Big Data Pipeline — what do you want to do?"
  echo "  1) demo        start AWS + show results (no new agent run)"
  echo "  2) full        run BOTH pipelines end to end"
  echo "  3) eu_sales    run the PostgreSQL pipeline"
  echo "  4) shop_orders run the MongoDB pipeline"
  echo "  5) results     show S3 / Trino / Grafana results"
  echo "  6) ui          open the Streamlit UI"
  echo "  7) check       offline tests + eval + security gate"
  echo "  8) pause       stop the database (save money)"
  echo "  9) status      AWS power state"
  read -r -p "Choose 1-9: " c
  case "$c" in
    1) set -- demo ;; 2) set -- full ;; 3) set -- eu_sales ;; 4) set -- shop_orders ;;
    5) set -- results ;; 6) set -- ui ;; 7) set -- check ;; 8) set -- pause ;; 9) set -- status ;;
    *) die "Unknown choice." ;;
  esac
  main "$@"
}

main() {
  local cmd="${1:-}" opt="${2:-}"
  case "$cmd" in
    "") menu ;;
    demo) load_env; start_aws; show_results ;;
    full)
      load_env; start_aws || exit 1
      yellow "Note: the free Gemini tier allows ~20 requests/day per model; two runs may not fit in one day."
      run_pipeline eu_sales "$opt" && run_pipeline shop_orders "$opt"
      show_results ;;
    eu_sales) load_env; start_aws || exit 1; run_pipeline eu_sales "$opt" && show_results ;;
    shop_orders) load_env; check_aws; run_pipeline shop_orders "$opt" && show_results ;;
    results) load_env; show_results ;;
    ui) load_env; run_ui ;;
    check) run_checks ;;
    status) load_env; make -s aws-status ;;
    pause) load_env; make -s aws-pause ;;
    destroy) load_env; confirm_destroy ;;
    -h|--help|help) sed -n '2,17p' "$0" ;;
    *) die "Unknown command '$cmd'. Run '$0 help'." ;;
  esac
}

main "$@"
