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
  "providers": [ { "name":"...","phone":"...","details":"... Please verify contact details independently.","address":"...","location_note":"EXACT|GENERAL|NEARBY","confidence":"HIGH|MEDIUM|LOW" } ],
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
    "pakistan": "+92", "peshawar": "+92", "khyber pakhtunkhwa": "+92",
    "karachi": "+92", "lahore": "+92", "islamabad": "+92", "rawalpindi": "+92",
    # India
    "india": "+91", "mumbai": "+91", "delhi": "+91",
    # USA
    "united states": "+1", "usa": "+1", "new york": "+1", "los angeles": "+1", "texas": "+1",
    # UK
    "united kingdom": "+44", "uk": "+44", "london": "+44",
    # UAE
    "united arab emirates": "+971", "uae": "+971", "dubai": "+971", "abu dhabi": "+971",
    # Morocco
    "morocco": "+212", "casablanca": "+212",
}

def _infer_country_code(location_hint: str) -> str | None:
    if not location_hint:
        return None
    s = location_hint.lower()
    for k, cc in _COUNTRY_HINTS.items():
        if k in s:
            return cc
    return None

def _normalize_phone_e164(raw: str, location_hint: str = "") -> str:
    """Best-effort E.164 formatting using simple rules + location hint."""
    if not raw:
        return ""
    s = raw.strip()
    # Keep only + and digits first
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

    # Apply simple rules for common cases
    if cc:
        # If already starts with CC digits, just add '+'
        if (cc == "+92" and digits.startswith("92")) or \
           (cc == "+91" and digits.startswith("91")) or \
           (cc == "+971" and digits.startswith("971")) or \
           (cc == "+44" and digits.startswith("44")):
            return "+" + digits
        if cc == "+1":
            # 1XXXXXXXXXX (11) or XXXXXXXXXX (10)
            if len(digits) == 11 and digits.startswith("1"):
                return "+" + digits
            if len(digits) == 10:
                return "+1" + digits
        # Common local leading 0
        if digits.startswith("0"):
            digits = digits.lstrip("0")
        return cc + digits

    # Fallback: no hint; prefix '+'
    return "+" + digits

def extract_explicit_location_from_query(query: str) -> str | None:
    """
    Extracts an explicit user-provided location (city/region/country) from the prompt.
    We treat 'near me/around me/here' as NOT an explicit location.
    """
    if not query:
        return None
    q = " ".join(query.strip().split())
    # If the query contains near-me phrases, we won't consider that as explicit location
    if requests_frontend_location(q):
        # It still might also contain 'in Peshawar', so we try to capture a concrete place
        pass
    # Capture text after common prepositions
    m = re.search(r"\b(?:in|at|from|within|around|near)\s+([A-Za-z][A-Za-z .,'\-]{2,60})", q, flags=re.I)
    if m:
        candidate = m.group(1).strip().strip(",.")
        # Ignore if candidate itself is a near-me phrase
        if not re.search(r"\b(near\s*me|around\s*me|here)\b", candidate, flags=re.I):
            return candidate
    # Also allow simple trailing place with no preposition, e.g., "plumber Peshawar"
    m2 = re.search(r"\b([A-Za-z][A-Za-z .,'\-]{2,60})$", q, flags=re.I)
    if m2 and not re.search(r"\b(near\s*me|around\s*me|here)\b", m2.group(1), flags=re.I):
        tail = m2.group(1).strip().strip(",.")
        # Heuristic: avoid capturing generic words
        if len(tail.split()) <= 5 and not re.search(r"\b(plumber|electrician|carpenter|mechanic|doctor|dentist|lawyer|cleaner|painter|roofer|locksmith|hvac|pest|service|services)\b", tail, flags=re.I):
            return tail
    return None

# ---------- Core logic ----------

def detect_provider_count(query: str) -> int:
    q = (query or "").lower()
    if "load more" in q or "more providers" in q:
        return 10
    if "all" in q or "maximum" in q:
        return 20
    m = re.search(r"\b(\d+)\b", q)
    if m:
        try:
            n = int(m.group(1))
            return min(max(n, 3), 20)
        except Exception:
            return 5
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
            "details": "This entry was added because web results were insufficient. Please verify contact details independently.",
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
            "message": "Web search tool unavailable or parse step failed — cannot return real web-sourced providers.",
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
        mloc = re.search(r"\b(?:in|at|from|around|near)\s+([A-Za-z0-9 .,\-']{2,60})", (query or ""), flags=re.I)
        loc = mloc.group(1).strip().strip(".,") if mloc else None
        parsed = {
            "ai_data": {
                "intent": "find_service" if svc else None,
                "service": svc,
                "location": loc,
                "confidence": 0.6 if svc else 0.0,
                "count": provider_count
            },
            "state": "complete" if svc and loc else ("need_service" if not svc else "need_location"),
            "message": "",
            "suggestions": ["plumber", "electrician"] if not svc else [],
            "providers": []
        }

    ai_data = parsed.setdefault("ai_data", {})
    ai_data["count"] = int(ai_data.get("count") or provider_count)

    # -------- Location precedence (explicit > frontend > parsed/none) --------
    explicit_loc = extract_explicit_location_from_query(query)
    near_me_flag = requests_frontend_location(query)

    if explicit_loc:
        # If user explicitly said a place, ALWAYS use it
        ai_data["location"] = explicit_loc
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
        used_frontend = False
    elif near_me_flag and frontend_location:
        # No explicit place; user said near me/around me -> use frontend
        ai_data["location"] = frontend_location
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
        used_frontend = True
    elif ai_data.get("location"):
        # Use whatever the parser extracted
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
        used_frontend = False
    else:
        parsed["state"] = "need_location"
        ai_data["location"] = None
        used_frontend = False

    # Backfill from past assistant turn only if still missing and we did NOT use frontend
    if not ai_data.get("location") and not used_frontend:
        with conversation_lock:
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    try:
                        past = json.loads(msg["content"])
                        past_ai = past.get("ai_data", {})
                        if not ai_data.get("service") and past_ai.get("service"):
                            ai_data["service"] = past_ai.get("service")
                        if not ai_data.get("location") and past_ai.get("location"):
                            ai_data["location"] = past_ai.get("location")
                    except Exception:
                        pass
                    break

    parsed["ai_data"] = ai_data

    if parsed.get("state") == "complete":
        service = ai_data.get("service")
        location = ai_data.get("location")
        desired = int(ai_data.get("count") or provider_count)

        provider_prompt = (
            f"Using up-to-date web search results, return ONLY valid JSON with key 'providers' containing exactly {desired} unique real providers.\n"
            f"Service: {service}\nLocation: {location}\n\n"
            "Rules:\n"
            " - Only include providers physically located in or clearly servicing the specified location (city/region/country).\n"
            " - Do not include providers from other cities/countries.\n"
            " - Assume common geography (e.g., 'Peshawar' -> Pakistan; 'Mumbai' -> India; 'Texas' -> United States).\n\n"
            "Each provider object must have keys: name, phone, details, address, location_note (EXACT|GENERAL), confidence (HIGH|MEDIUM|LOW).\n"
            "Requirements:\n"
            " - Use ONLY information directly verifiable on live web pages (the web search tool will be used).\n"
            " - If a field is unavailable, set it to an empty string.\n"
            " - Ensure each 'details' ends with 'Please verify contact details independently.'\n\n"
            "Return exactly one top-level JSON object and nothing else."
        )

        prov_resp_obj, prov_raw, prov_tool = try_web_response(provider_prompt, model="gpt-4o", max_output_tokens=1500, temperature=0.1)

        if not prov_resp_obj or not prov_raw:
            result = {
                "valid": False,
                "message": "Provider generation failed or web search tool unavailable — cannot return real providers.",
                "state": "error",
                "providers": [],
                "suggestions": [],
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
                "message": "Provider generation returned unparsable or missing 'providers' — aborting to avoid fabricated data.",
                "state": "error",
                "providers": [],
                "suggestions": [],
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
                "message": f"Here are {len(providers)} providers we found in {location}. Please verify details independently.",
                "state": "complete",
                "providers": providers,
                "suggestions": [service, "repair"],
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
    if not suggestions and ai_data.get("service"):
        suggestions = [ai_data["service"], "repair"]  # fallback keywords

    if isinstance(parsed, dict):
        default_msg = f"Here are {ai_data.get('count')} {ai_data.get('service','')} providers in {ai_data.get('location','your area')}."
        message = parsed.get("message", default_msg)
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
            "provider_tool_used": prov_resp_obj and getattr(prov_resp_obj, "model", None),
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
