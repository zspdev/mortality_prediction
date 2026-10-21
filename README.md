# mortality_prediction

Estimates how COVID-19 in-hospital mortality (deaths per admission) rises with hospital load, using hospital census, ICU and death series from BE, BR-Rio, ES, HU, IT and TR. The output is a load multiplier on hospital fatality for tuning an agent-based model (PanSIM) of how the death rate changes with healthcare-worker availability.

See `PLAN.md` for the target definition, phases and open questions, and `docs/` for data provenance.

Setup: `uv sync`; notebooks with `uv run jupyter lab`.
