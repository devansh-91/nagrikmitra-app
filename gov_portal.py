"""
Government grievance portal adapter interface.

There is no confirmed public submission API for CPGRAMS (pgportal.gov.in,
the DARPG-run national grievance portal) or for the state/municipal portals
this project's taxonomy maps to (PWD, Jal Board, Discom, etc.). Their
citizen-facing forms require CAPTCHA and mobile OTP verification, which
exist specifically to stop automated/bulk filing. Nagarik Mitra therefore
does NOT scrape or drive those forms — doing so would mean filing real
grievances into a live government system without a verifiable, authorized
integration, which is out of scope for what is explicitly a B.Tech CSE
(AI/ML) demo project (see README.md).

This module defines the seam a *real* integration would plug into, so that
if you obtain an actual, documented API credential from a portal operator
(e.g. a formal DARPG/CPGRAMS API partnership, or a municipal corporation's
e-governance API), you implement `GovPortalAdapter` against that real API
and swap it in for `MockGovPortalAdapter` in mcp_server.py. Until then,
MockGovPortalAdapter simulates the round trip (deterministic fake reference
number, no network call, nothing leaves this machine) so the rest of the
pipeline — MCP tool, ticket lookup, response shape — can be built and tested
end-to-end without pretending to talk to a live government system.
"""
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone

# One simulated portal per domain (mirrors config/taxonomy.yaml's department
# mapping), so the demo shows "each department has its own portal" rather
# than funnelling everything through one name. "Other / Unknown" maps to the
# national portal, matching how a real low-confidence/ambiguous grievance
# would fall back to CPGRAMS rather than a specific department system.
PORTAL_BY_DOMAIN = {
    "Roads": "e-PWD Grievance Portal",
    "Lights": "Municipal Corporation e-Seva",
    "Water": "Jal Board Grievance Portal",
    "Waste": "SWM Grievance Portal",
    "Sanitation": "Sewerage Board Grievance Portal",
    "Electricity": "Discom Consumer Portal",
    "Health": "District Health Portal",
    "Other / Unknown": "CPGRAMS (National Portal)",
}
DEFAULT_PORTAL = "CPGRAMS (National Portal)"

# Short alphanumeric code per portal, used only in the simulated reference
# number (a real portal's own ids are its own format, not this one).
PORTAL_CODE = {
    "e-PWD Grievance Portal": "EPWD",
    "Municipal Corporation e-Seva": "MCSEVA",
    "Jal Board Grievance Portal": "JALBRD",
    "SWM Grievance Portal": "SWM",
    "Sewerage Board Grievance Portal": "SEWBRD",
    "Discom Consumer Portal": "DISCOM",
    "District Health Portal": "HEALTH",
    "CPGRAMS (National Portal)": "CPGRAMS",
}


def portal_for_domain(domain: str) -> str:
    """Simulated portal name for a domain. None of these are real endpoints
    — see this module's docstring."""
    return PORTAL_BY_DOMAIN.get(domain, DEFAULT_PORTAL)


class PortalSubmissionError(Exception):
    """Raised when a government portal adapter cannot submit a ticket."""


class GovPortalAdapter(ABC):
    """Interface a real government-portal integration must implement.

    A real implementation would need, at minimum: an authenticated HTTP
    client for the portal's actual (documented, sanctioned) API, a mapping
    from Nagarik Mitra's taxonomy (config/taxonomy.yaml) to that portal's own
    category/department codes, and error handling for whatever failure modes
    that API defines (auth expiry, rate limits, validation errors, etc.).
    None of that exists here because no such API access has been confirmed.
    """

    @abstractmethod
    def submit(self, ticket: dict, portal: str = None) -> dict:
        """Submit `ticket` (a row from nagrikmitra.store, i.e. a dict with at
        least text/domain/priority/department/name/location) to the named
        external portal. Returns a dict describing the outcome; raises
        PortalSubmissionError on failure."""
        raise NotImplementedError


class MockGovPortalAdapter(GovPortalAdapter):
    """Simulates a government portal submission for demos/testing.

    Makes no network call and contacts no real system. Produces a
    deterministic, clearly-fake reference number so it can never be mistaken
    for a genuine CPGRAMS/portal registration number.
    """

    def submit(self, ticket: dict, portal: str = None) -> dict:
        if not ticket.get("text"):
            raise PortalSubmissionError("ticket has no complaint text to submit")

        portal = portal or portal_for_domain(ticket.get("domain"))
        code = PORTAL_CODE.get(portal, "GOV")
        digest = hashlib.sha256(
            f"{ticket.get('id')}|{ticket.get('text')}|{portal}".encode("utf-8")
        ).hexdigest()[:10].upper()

        return {
            "submitted": True,
            "simulated": True,
            "portal": portal,
            "portal_reference": f"MOCK-{code}-{digest}",
            "note": (
                "Simulated only — no live government portal was contacted. "
                "Replace MockGovPortalAdapter with a real GovPortalAdapter "
                "implementation once genuine, authorized API access to this "
                "portal is available."
            ),
            "submitted_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "ticket_id": ticket.get("id"),
            "department": ticket.get("department"),
        }
