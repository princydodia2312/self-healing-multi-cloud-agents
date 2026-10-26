# --- Variables ---
PYTHON = uv run python
APP = main.py
SCRIPTS_DIR = scripts

## Single source of truth for dependencies: pyproject.toml
## Before running: cp .env.example .env  (then fill in your credentials)

# --- Setup ---

.PHONY: install
install: ## Install all dependencies using uv
	@echo "Installing dependencies..."
	uv sync

# --- AWS baseline infrastructure (one-time) ---

.PHONY: bootstrap-aws
bootstrap-aws: ## Provision baseline AWS infrastructure (EKS + RDS + S3 + ECR + SSM)
	@echo "Provisioning AWS baseline infrastructure..."
	terraform -chdir=bootstrap/aws init
	terraform -chdir=bootstrap/aws apply
	@echo "Exporting bootstrap outputs -> .bootstrap_outputs.json (used by the NL/Streamlit deploy path)..."
	$(PYTHON) scripts/export_bootstrap_outputs.py aws

.PHONY: destroy-aws
destroy-aws: ## DANGER: Tear down all baseline AWS infrastructure
	@echo "Destroying baseline AWS infrastructure..."
	terraform -chdir=bootstrap/aws init
	terraform -chdir=bootstrap/aws destroy

.PHONY: aws-pause
aws-pause: ## Stop RDS + scale EKS nodes to 0 to cut cost between sessions (keeps everything)
	@bash bootstrap/aws/power.sh pause

.PHONY: aws-resume
aws-resume: ## Start RDS + EKS nodes to resume work
	@bash bootstrap/aws/power.sh resume

.PHONY: aws-status
aws-status: ## Show AWS baseline power state (RDS + EKS node groups)
	@bash bootstrap/aws/power.sh status

.PHONY: cost
cost: ## Print the monthly AWS baseline cost estimate (deterministic list prices)
	$(PYTHON) -m utils.cost_estimator

# --- Pipeline demo ---

.PHONY: ingest
ingest: ## Sync the Knowledge Base (engineering standards) to Pinecone for RAG
	@echo "Syncing standards to Pinecone..."
	$(PYTHON) $(SCRIPTS_DIR)/ingest_to_pinecone.py --path knowledge_base

.PHONY: chaos
chaos: ## Seed dirty data into the source DB. Usage: make chaos [db_type=postgres|sqlite] [rows=100]
	@echo "Seeding chaos into eu_sales (db_type=$(if $(db_type),$(db_type),postgres))..."
	$(PYTHON) $(SCRIPTS_DIR)/seed_chaos.py --target eu_sales --db-type $(if $(db_type),$(db_type),postgres) --rows $(if $(rows),$(rows),100)

.PHONY: run
run: ## Run the Self-Healing Agent. Usage: make run [p=eu_sales]
	@echo "Starting Self-Healing Agent for: $(if $(p),$(p),eu_sales)..."
	$(PYTHON) $(APP) $(if $(p),$(p),eu_sales)

.PHONY: demo
demo: ## Full AWS demo: ingest standards + seed chaos + run eu_sales
	@make ingest
	@make chaos
	@make run p=eu_sales

.PHONY: streamlit ui
streamlit ui: ## Launch the Streamlit UI → http://localhost:8501
	uv run streamlit run streamlit_app.py --server.fileWatcherType none

# --- Quality ---

.PHONY: test
test: ## Run the test suite
	uv run pytest tests/ -v --tb=short

.PHONY: lint
lint: ## Lint the whole repo with ruff (same command CI runs)
	uv run ruff check .

.PHONY: format
format: ## Autofix lint findings with ruff
	uv run ruff check --fix .

.PHONY: eval-replay
eval-replay: ## Self-heal eval (OFFLINE — no cloud/LLM): score the failure corpus, regenerate evals/report/
	$(PYTHON) -m evals.harness.runner

.PHONY: eval-check
eval-check: ## CI parity: fail if evals/report is stale OR routing/evidence-gate regressed vs the corpus
	$(PYTHON) -m evals.harness.runner --check
	$(PYTHON) -m evals.harness.runner --gate

.PHONY: eval-live
eval-live: ## Medic diagnosis eval (LIVE — needs an LLM key). Override with model=<id>
	$(PYTHON) -m evals.harness.eval_live $(if $(model),--model $(model),)

.PHONY: heal
heal: ## Route + validate a failing CI log through the real Medic logic (offline). Usage: make heal LOG=run.log
	@$(if $(LOG),,$(error Error: LOG is undefined. Usage: make heal LOG=run.log))
	$(PYTHON) -m evals.heal --log $(LOG)

.PHONY: security-gate
security-gate: ## Security gate over the generated bundle (k8s/, Dockerfile, .github/workflows/pipe_*)
	$(PYTHON) -m policy.security_analyzer .

.PHONY: gate-proof
gate-proof: ## Prove the security gate REFUSES the unsafe fixtures + PASSES the clean AWS goldens
	$(PYTHON) -m pytest tests/test_security_gate.py -v

.PHONY: clean
clean: ## Remove temporary files and caches
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .uv_cache pipeline_execution.log

.PHONY: help
help: ## Display this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
