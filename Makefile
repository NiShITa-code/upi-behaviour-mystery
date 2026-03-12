.PHONY: run test lint pipeline clean

# Run the interactive dashboard
run:
	streamlit run app.py

# Run full test suite with coverage
test:
	cd upi_project && pytest tests/ -v --cov=src --cov-report=term-missing

# Run linter
lint:
	cd upi_project && ruff check src/ tests/

# Run the pipeline CLI (full analysis, no UI)
pipeline:
	cd upi_project && python -m src.pipeline

pipeline-small:
	cd upi_project && python -m src.pipeline --n-users 2000

# Clean generated artifacts
clean:
	rm -rf upi_project/artifacts/*.joblib upi_project/artifacts/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
