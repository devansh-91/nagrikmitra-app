"""
Domain -> Department and Priority -> SLA lookups.

Responsibilities:
- Load config/taxonomy.yaml at import time.
- Fall back to an identical in-code dict if the YAML is missing/unreadable
  (never crash the app on a missing config file).
- Expose: DOMAINS, DEPARTMENTS (domain -> full/short name), PRIORITIES,
  SLA_HOURS (priority -> hours), get_department(domain), get_sla_hours(priority).
- Taxonomy is FROZEN for v1 — do not add/remove classes here.
"""
import yaml

from nagrikmitra.config import TAXONOMY_YAML_PATH

# Fallback taxonomy — must stay identical to config/taxonomy.yaml.
_FALLBACK_TAXONOMY = {
    "domains": {
        "Roads": {
            "department_full": "PWD (Public Works Department)",
            "department_short": "PWD",
        },
        "Lights": {
            "department_full": "Municipal Corporation — Street Lighting Division",
            "department_short": "Municipal Corp.",
        },
        "Water": {
            "department_full": "Jal Board (Water Supply and Sewerage)",
            "department_short": "Jal Board",
        },
        "Waste": {
            "department_full": "Solid Waste Management Department",
            "department_short": "SWM Dept.",
        },
        "Sanitation": {
            "department_full": "Sewerage and Drainage Board",
            "department_short": "Sewerage Board",
        },
        "Electricity": {
            "department_full": "Discom (Power Distribution Company)",
            "department_short": "Discom",
        },
        "Health": {
            "department_full": "District Health Department",
            "department_short": "Health Dept.",
        },
        "Other / Unknown": {
            "department_full": "General Administration (human review)",
            "department_short": "Human review",
        },
    },
    "priorities": {
        "High": {
            "meaning": "danger, sparking, overflow into home, dengue cluster, accident, outage + safety risk",
            "sla_hours": 48,
        },
        "Medium": {
            "meaning": "leak, blocked drain, delayed collection, fluctuations",
            "sla_hours": 168,
        },
        "Low": {
            "meaning": "suggestion, minor crack, FYI",
            "sla_hours": 336,
        },
    },
    "routing": {
        "domain_margin_threshold": 0.08,
        "abstain_threshold": 0.45,
    },
}


def _load_taxonomy() -> dict:
    try:
        with open(TAXONOMY_YAML_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if not loaded or "domains" not in loaded or "priorities" not in loaded:
            raise ValueError("taxonomy.yaml missing required sections")
        return loaded
    except (OSError, yaml.YAMLError, ValueError):
        return _FALLBACK_TAXONOMY


_TAXONOMY = _load_taxonomy()

DOMAINS = list(_TAXONOMY["domains"].keys())
DEPARTMENTS = _TAXONOMY["domains"]
PRIORITIES = list(_TAXONOMY["priorities"].keys())
SLA_HOURS = {p: v["sla_hours"] for p, v in _TAXONOMY["priorities"].items()}

OTHER_DOMAIN = "Other / Unknown"


def get_department(domain: str, short: bool = False) -> str:
    """Return the department name for a domain; unknown domains fall back to human review."""
    entry = DEPARTMENTS.get(domain, DEPARTMENTS[OTHER_DOMAIN])
    return entry["department_short"] if short else entry["department_full"]


def get_sla_hours(priority: str) -> int:
    """Return SLA hours for a priority; unknown priorities default to Low's SLA."""
    return SLA_HOURS.get(priority, SLA_HOURS["Low"])
