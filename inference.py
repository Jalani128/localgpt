# inference.py  -- Responses API + web_search enforced (fail if web unavailable)
from openai import OpenAI
import json
import threading
import re
from datetime import datetime
from config import OPENAI_API_KEY
import logging
import traceback

client = OpenAI(api_key=OPENAI_API_KEY)

conversation = [
    {
        "role": "system",
        "content": """You are a helpful service finder AI. CRITICAL: You MUST ALWAYS return valid JSON in the exact format specified below.

Extract: service type, location, intent from user input.
Respond based on what you have:
- Both service+location: Find 5-8 realistic providers, state="complete"
- Missing service: Ask what service they need, state="need_service"
- Missing location: Ask where they're located, state="need_location"
- Greeting/chat: Be friendly, redirect to services, state="redirect"
- Unclear: Ask for clarification, state="error"

MANDATORY: Return ONLY valid JSON in this EXACT format (no extra text):
{
  "valid": true,
  "message": "...",
  "state": "complete|need_service|need_location|redirect|error",
  "providers": [ { "name":"...","phone":"...","details":"...","address":"...","location_note":"EXACT|GENERAL|NEARBY","confidence":"HIGH|MEDIUM|LOW","service":"..." } ],
  "suggestions": ["..."],
  "ai_data": {"intent":"...","service":"...","location":"...","confidence":0.9},
  "usage_report": {}
}"""
    }
]

conversation_lock = threading.Lock()

logger = logging.getLogger("localgpt2.inference")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(_h)

# ---------- Phone & location helpers ----------

_COUNTRY_HINTS = {
    # Pakistan + major cities/regions
    "pakistan": "+92", "peshawar": "+92", "peshaw": "+92", "hayatabad": "+92",
    "khyber": "+92", "kpk": "+92", "khyber pakhtunkhwa": "+92",
    "karachi": "+92", "lahore": "+92", "islamabad": "+92", "rawalpindi": "+92",
    # India
    "india": "+91", "mumbai": "+91", "delhi": "+91", "bangalore": "+91", "kolkata": "+91", "chennai": "+91",
    # USA
    "united states": "+1", "usa": "+1", "new york": "+1", "los angeles": "+1", "texas": "+1", "california": "+1",
    "washington": "+1", "chicago": "+1",
    # UK
    "united kingdom": "+44", "uk": "+44", "london": "+44", "manchester": "+44",
    # UAE
    "united arab emirates": "+971", "uae": "+971", "dubai": "+971", "abu dhabi": "+971", "sharjah": "+971",
    # Morocco
    "morocco": "+212", "casablanca": "+212", "rabat": "+212",
    # Saudi Arabia
    "saudi arabia": "+966", "ksa": "+966", "riyadh": "+966", "jeddah": "+966",
}

_COUNTRY_NAME_BY_KEY = {
    # Pakistan
    "pakistan": "Pakistan", "peshawar": "Pakistan", "peshaw": "Pakistan", "hayatabad": "Pakistan",
    "khyber": "Pakistan", "kpk": "Pakistan", "khyber pakhtunkhwa": "Pakistan",
    "karachi": "Pakistan", "lahore": "Pakistan", "islamabad": "Pakistan", "rawalpindi": "Pakistan",
    # India
    "india": "India", "mumbai": "India", "delhi": "India", "bangalore": "India", "kolkata": "India", "chennai": "India",
    # USA
    "united states": "United States", "usa": "United States", "new york": "United States", "los angeles": "United States",
    "texas": "United States", "california": "United States", "washington": "United States", "chicago": "United States",
    # UK
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "london": "United Kingdom", "manchester": "United Kingdom",
    # UAE
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates",
    "dubai": "United Arab Emirates", "abu dhabi": "United Arab Emirates", "sharjah": "United Arab Emirates",
    # Morocco
    "morocco": "Morocco", "casablanca": "Morocco", "rabat": "Morocco",
    # Saudi Arabia
    "saudi arabia": "Saudi Arabia", "ksa": "Saudi Arabia", "riyadh": "Saudi Arabia", "jeddah": "Saudi Arabia",
}

def _infer_country_code(location_hint: str) -> str | None:
    if not location_hint:
        return None
    s = location_hint.lower()
    for k, cc in _COUNTRY_HINTS.items():
        if k in s:
            return cc
    # extra fuzzy guards
    if "peshaw" in s or "khyber" in s or "kpk" in s:
        return "+92"
    if "mumba" in s or "delh" in s:
        return "+91"
    return None

def _infer_country_name(location_hint: str) -> str | None:
    if not location_hint:
        return None
    s = location_hint.lower()
    for k, name in _COUNTRY_NAME_BY_KEY.items():
        if k in s:
            return name
    if "peshaw" in s or "khyber" in s or "kpk" in s:
        return "Pakistan"
    if "mumba" in s or "delh" in s:
        return "India"
    return None

def _normalize_phone_e164(raw: str, location_hint: str = "") -> str:
    """Best-effort E.164 formatting using simple rules + location hint."""
    if not raw:
        return ""
    s = raw.strip()
    kept = re.sub(r"[^\d+]", "", s)

    # Already +CC...
    if kept.startswith("+"):
        return "+" + re.sub(r"[^\d]", "", kept)

    # 00CC...
    if kept.startswith("00"):
        return "+" + re.sub(r"[^\d]", "", kept[2:])

    # Bare digits
    digits = re.sub(r"\D", "", kept)
    if not digits:
        return ""

    cc = _infer_country_code(location_hint)

    if cc:
        if (cc == "+92" and digits.startswith("92")) or \
           (cc == "+91" and digits.startswith("91")) or \
           (cc == "+971" and digits.startswith("971")) or \
           (cc == "+44" and digits.startswith("44")):
            return "+" + digits
        if cc == "+1":
            if len(digits) == 11 and digits.startswith("1"):
                return "+" + digits
            if len(digits) == 10:
                return "+1" + digits
        if digits.startswith("0"):
            digits = digits.lstrip("0")
        return cc + digits

    return "+" + digits

def extract_explicit_location_from_query(query: str) -> str | None:
    """
    Extract an explicit user-provided location (city/region/country) from the prompt.
    'near me/around me/here' is NOT an explicit location.
    """
    if not query:
        return None
    q = " ".join(query.strip().split())

    # Try "in/at/from/within/around/near <Place>"
    m = re.search(r"\b(?:in|at|from|within|around|near)\s+([A-Za-z][A-Za-z .,'\-]{2,60})", q, flags=re.I)
    if m:
        candidate = m.group(1).strip().strip(",.")
        if not re.search(r"\b(near\s*me|around\s*me|here)\b", candidate, flags=re.I):
            return candidate

    # Also allow final token heuristic: "... plumber Peshawar"
    m2 = re.search(r"\b([A-Za-z][A-Za-z .,'\-]{2,60})$", q, flags=re.I)
    if m2 and not re.search(r"\b(near\s*me|around\s*me|here)\b", m2.group(1), flags=re.I):
        tail = m2.group(1).strip().strip(",.")
        if len(tail.split()) <= 5 and not re.search(
            r"\b(plumber|electrician|carpenter|mechanic|doctor|dentist|lawyer|cleaner|painter|roofer|locksmith|hvac|pest|service|services)\b",
            tail, flags=re.I):
            return tail
    return None

def extract_explicit_count(query: str) -> int | None:
    """Return a user-typed number (3..20) if present; otherwise None."""
    if not query:
        return None
    m = re.search(r"\b(\d{1,2})\b", query)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return min(max(n, 3), 20)
    except Exception:
        return None

# ---------- Core logic ----------

def detect_provider_count(query: str) -> int:
    # Default to 5; only change if the user typed a number.
    m = re.search(r"\b(\d{1,2})\b", (query or ""))
    if m:
        try:
            n = int(m.group(1))
            return min(max(n, 3), 20)
        except Exception:
            pass
    return 5

def safe_json_parse(text: str) -> dict:
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {}
    return {}

NEAR_ME_PHRASES = [
    "near me", "nearby", "my area", "around here", "around me", "here in the area",
    "here", "near us", "nearby us", "close to me", "close by", "in my area",
    "newar me", "newar", "aroundme", "nearme"
]
_NEAR_ME_PATTERNS = [re.compile(r"\b" + re.escape(p) + r"\b", flags=re.I) for p in NEAR_ME_PHRASES]

def requests_frontend_location(query: str) -> bool:
    if not query:
        return False
    for pat in _NEAR_ME_PATTERNS:
        if pat.search(query):
            return True
    return False

def normalize_provider(p: dict, default_service: str = "", location_hint: str = "") -> dict:
    """Normalize provider to required schema + add 'service' and format phone to E.164."""
    phone_norm = _normalize_phone_e164(str(p.get("phone", "")).strip(), location_hint)
    return {
        "name": str(p.get("name", "")).strip(),
        "phone": phone_norm,
        "details": str(p.get("details", "")).strip(),
        "address": str(p.get("address", "")).strip(),
        "location_note": str(p.get("location_note", "GENERAL")).strip(),
        "confidence": str(p.get("confidence", "LOW")).strip(),
        "service": str(p.get("service", default_service) or "").strip(),
    }

def enforce_exact_count(providers: list, desired: int, default_service: str = "", location_hint: str = "") -> list:
    """Ensure we have exactly `desired` providers; preserve schema and add placeholders if needed."""
    if not isinstance(providers, list):
        providers = []
    normalized = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        entry = normalize_provider(p, default_service=default_service, location_hint=location_hint)
        normalized.append(entry)
    if len(normalized) >= desired:
        return normalized[:desired]
    padded = normalized.copy()
    for i in range(len(normalized), desired):
        padded.append({
            "name": f"Provider {i+1} (incomplete)",
            "phone": "",
            "details": "Added because web results were limited.",
            "address": "",
            "location_note": "GENERAL",
            "confidence": "LOW",
            "service": default_service,
        })
    return padded

def try_web_response(prompt_text: str, model: str = "gpt-4o", max_output_tokens: int = 800, temperature: float = 0.0):
    attempts = []
    for tool_type in ("web_search", "web_search_preview"):
        try:
            web_tool = {"type": tool_type, "search_context_size": "high"}
            resp = client.responses.create(
                model=model,
                tools=[web_tool],
                input=[{"role": "user", "content": prompt_text}],
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
            raw = getattr(resp, "output_text", None)
            if not raw:
                parts = []
                for item in getattr(resp, "output", []) or []:
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
                raw = "\n".join(parts)
            return resp, raw, tool_type
        except Exception as e:
            attempts.append((tool_type, repr(e)))
            logger.debug("try_web_response attempt failed for %s: %s", tool_type, e)
            continue
    logger.warning("try_web_response: all attempts failed: %s", attempts)
    return None, None, None

def process_query(query: str, frontend_location: str = None) -> dict:
    global conversation
    logger.info("process_query received query: %s", query)
    provider_count = detect_provider_count(query)

    with conversation_lock:
        if len(conversation) > 7:
            conversation = [conversation[0]] + conversation[-6:]
        conversation.append({"role": "user", "content": query})

    parse_prompt = (
        "Use live web search results to extract and return ONLY valid JSON with keys: "
        "ai_data, state, message, suggestions, providers. "
        "ai_data must contain: intent, service, location, confidence, count. "
        f"User: {query}\n"
        "Return ONE JSON object only."
    )

    parse_resp_obj, parse_raw, parse_tool = try_web_response(parse_prompt, model="gpt-4o", max_output_tokens=800, temperature=0.0)

    parsed = {}
    if not parse_resp_obj or not parse_raw:
        result = {
            "valid": False,
            "message": "I can help. What service and city are you looking for?",
            "state": "error",
            "providers": [],
            "suggestions": [],
            "ai_data": {"intent": None, "service": None, "location": None, "confidence": 0.0},
            "usage_report": {
                "error": "web_tool_unavailable_or_parse_failed",
                "attempted_tools": ["web_search", "web_search_preview"],
                "timestamp": datetime.now().isoformat()
            }
        }
        try:
            with conversation_lock:
                conversation.append({"role": "assistant", "content": json.dumps(result)})
        except Exception:
            pass
        return result

    parsed = safe_json_parse(parse_raw)
    if not isinstance(parsed, dict) or not parsed.get("ai_data"):
        svc = None
        msvc = re.search(r"\b(plumber|electrician|carpenter|mechanic|doctor|dentist|lawyer|cleaner|painter|roofer|locksmith|hvac|pest)\b", (query or "").lower())
        if msvc:
            svc = msvc.group(1)
        # We do not trust parser for location; explicit or frontend only
        parsed = {
            "ai_data": {
                "intent": "find_service" if svc else None,
                "service": svc,
                "location": None,
                "confidence": 0.6 if svc else 0.0,
                "count": provider_count
            },
            "state": "need_location" if svc else "need_service",
            "message": "",
            "suggestions": ["plumber", "electrician"] if not svc else [],
            "providers": []
        }

    ai_data = parsed.setdefault("ai_data", {})

    # ----- FORCE count to 5 unless user explicitly typed a number -----
    explicit_count = extract_explicit_count(query)
    if explicit_count is not None:
        ai_data["count"] = explicit_count
    else:
        ai_data["count"] = 5

    # -------- Location precedence (explicit > frontend > need_location) --------
    explicit_loc = extract_explicit_location_from_query(query)
    near_me_flag = requests_frontend_location(query)

    used_frontend = False
    if explicit_loc:
        ai_data["location"] = explicit_loc
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
    elif near_me_flag and frontend_location:
        ai_data["location"] = frontend_location
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
        used_frontend = True
    else:
        # Ask for location if none provided
        if not ai_data.get("location"):
            ai_data["location"] = None
            parsed["state"] = "need_location"

    # Backfill ONLY service from last assistant turn (NEVER location)
    with conversation_lock:
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                try:
                    past = json.loads(msg["content"])
                    past_ai = past.get("ai_data", {})
                    if not ai_data.get("service") and past_ai.get("service"):
                        ai_data["service"] = past_ai.get("service")
                except Exception:
                    pass
                break

    parsed["ai_data"] = ai_data

    if parsed.get("state") == "complete":
        service = ai_data.get("service")
        location = ai_data.get("location")
        desired = int(ai_data.get("count") or 5)
        country_name = _infer_country_name(location) or "Unknown"
        cc_hint = _infer_country_code(location) or ""

        provider_prompt = (
            f"Using up-to-date web search, return ONLY valid JSON with key 'providers' containing exactly {desired} unique real providers.\n"
            f"Service: {service}\n"
            f"Location: {location}\n"
            f"Inferred country: {country_name}\n\n"
            "Geography guardrails:\n"
            " - Results must be from the specified location/country. Do NOT include results from other countries.\n"
            " - If the country is Unknown, infer correctly from the city/region (e.g., Peshawar → Pakistan, Mumbai → India) and DO NOT assume United States unless the location is clearly in the US.\n"
            " - Prefer local sources/TLDs and listings that mention the city/region in the address.\n"
            " - Prefer phone numbers that match local numbering (e.g., country code patterns).\n\n"
            "Each provider object must have keys: name, phone, details, address, location_note (EXACT|GENERAL), confidence (HIGH|MEDIUM|LOW).\n"
            "Rules:\n"
            " - Use ONLY information visible on live web pages (the web search tool will be used).\n"
            " - If a field is unavailable, set it to an empty string.\n"
            " - Do NOT add any verification disclaimers to details.\n\n"
            "Return exactly one top-level JSON object and nothing else."
        )

        prov_resp_obj, prov_raw, prov_tool = try_web_response(provider_prompt, model="gpt-4o", max_output_tokens=1500, temperature=0.1)

        if not prov_resp_obj or not prov_raw:
            result = {
                "valid": False,
                "message": f"I couldn’t fetch providers for {service} in {location} right now.",
                "state": "error",
                "providers": [],
                "suggestions": [service or "service", (location or "nearby").split()[0]],
                "ai_data": ai_data,
                "usage_report": {
                    "error": "provider_generation_failed_or_no_web_tool",
                    "attempted_tools": ["web_search", "web_search_preview"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            try:
                with conversation_lock:
                    conversation.append({"role": "assistant", "content": json.dumps(result)})
            except Exception:
                pass
            return result

        new_data = safe_json_parse(prov_raw)
        if not isinstance(new_data, dict) or not isinstance(new_data.get("providers"), list):
            result = {
                "valid": False,
                "message": "I had trouble reading provider data. Mind trying again with the city name?",
                "state": "error",
                "providers": [],
                "suggestions": [service or "service", (location or "nearby").split()[0]],
                "ai_data": ai_data,
                "usage_report": {
                    "error": "provider_generation_unparseable",
                    "raw": prov_raw[:1000],
                    "timestamp": datetime.now().isoformat()
                }
            }
            try:
                with conversation_lock:
                    conversation.append({"role": "assistant", "content": json.dumps(result)})
            except Exception:
                pass
            return result

        # Check for 'source' on raw providers (normalizer drops unknown keys)
        raw_providers = new_data.get("providers", [])
        missing_source = any(not (isinstance(p, dict) and p.get("source")) for p in raw_providers)

        # Normalize to output schema; add service + E.164 phone
        providers = [normalize_provider(p, default_service=service, location_hint=location) for p in raw_providers]

        if missing_source:
            # Return providers anyway; log a warning & keep suggestions
            result = {
                "valid": True,
                "message": f"Here are {len(providers)} {service}s in {location}.",
                "state": "complete",
                "providers": providers,
                "suggestions": [service or "service", (location or "nearby").split()[0]],
                "ai_data": ai_data,
                "usage_report": {
                    "warning": "missing_sources_in_providers",
                    "raw_providers_sample": raw_providers[:3],
                    "timestamp": datetime.now().isoformat()
                }
            }
            try:
                with conversation_lock:
                    conversation.append({"role": "assistant", "content": json.dumps(result)})
            except Exception:
                pass
            return result

        parsed_providers = enforce_exact_count(providers, desired, default_service=service, location_hint=location)
        parsed["providers"] = parsed_providers
        parsed["state"] = "complete"
    else:
        parsed.setdefault("providers", [])
        parsed["state"] = parsed.get("state", "error")

    # Build friendly defaults and ensure suggestions are not empty
    ai_data = parsed.get("ai_data", {"intent": None, "service": None, "location": None, "confidence": 0.0})

    suggestions = parsed.get("suggestions", [])
    if not suggestions:
        if ai_data.get("service") and ai_data.get("location"):
            suggestions = [ai_data["service"], ai_data["location"].split()[0]]
        elif ai_data.get("service"):
            suggestions = [ai_data["service"], "nearby"]

    if isinstance(parsed, dict):
        if parsed.get("state") == "need_location":
            message = "Got it—what city should I look in?"
        elif parsed.get("state") == "need_service":
            message = "Sure—what kind of service do you need and where?"
        elif parsed.get("state") == "complete":
            message = f"Here are {ai_data.get('count')} {ai_data.get('service','')} providers in {ai_data.get('location','your area')}."
        else:
            message = "I’m ready—tell me the service and city."
    else:
        message = ""

    result = {
        "valid": True,
        "message": message,
        "state": parsed.get("state", "error"),
        "providers": parsed.get("providers", []),
        "suggestions": suggestions,
        "ai_data": ai_data,
        "usage_report": {
            "parse_tool_used": parse_resp_obj and getattr(parse_resp_obj, "model", None),
            "provider_tool_used": prov_resp_obj and getattr(prov_resp_obj, "model", None) if 'prov_resp_obj' in locals() else None,
            "timestamp": datetime.now().isoformat()
        }
    }

    try:
        with conversation_lock:
            conversation.append({"role": "assistant", "content": json.dumps(result)})
    except Exception:
        pass

    return result

if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        frontend_location = None
        result = process_query(query, frontend_location=frontend_location)
        print(json.dumps(result, indent=2, ensure_ascii=False))
