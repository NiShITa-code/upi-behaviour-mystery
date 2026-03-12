# Contributing to UPI Behaviour Mystery

Thanks for your interest in contributing! This document explains how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/upi-behaviour-mystery.git`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/your-feature`
5. Make your changes
6. Run tests: `pytest tests/ -v`
7. Commit with a conventional message: `feat(scope): description`
8. Push and open a Pull Request

## Development Setup

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src --cov-report=term-missing
```

All 59 tests should pass before submitting a PR.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(data): add new data loader` — new feature
- `fix(model): correct threshold calculation` — bug fix
- `test(uplift): add edge case tests` — new tests
- `docs: update installation guide` — documentation
- `chore: update dependencies` — maintenance

## Code Style

- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- No magic numbers — use `config/config.yaml`
- Data contracts via typed dataclasses
- Schema validation at data boundaries

## Testing

Tests live in `tests/` and use pytest. We care about **business-logic assertions**, not just smoke tests:

- Models must beat random (AUC > 0.5)
- Persuadables must have positive uplift
- Schema validation must reject bad data
- Simulation output must match NPCI calibration targets within tolerance

If you add a new module, add tests for it.

## Areas for Contribution

- **More uplift estimators** — S-Learner, X-Learner, doubly robust estimator
- **Real-world validation** — benchmarking against anonymized UPI data
- **Deployment** — Docker, Streamlit Cloud config, FastAPI wrapper
- **Visualization** — additional dashboard tabs or chart types
- **Documentation** — tutorials, blog posts, video walkthroughs

## Questions?

Open an issue and we'll get back to you.
