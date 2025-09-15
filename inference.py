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


def normalize_provider(p: dict) -> dict:
    return {
        "name": str(p.get("name", "")).strip(),
        "phone": str(p.get("phone", "")).strip(),
        "details": str(p.get("details", "")).strip(),
        "address": str(p.get("address", "")).strip(),
        "location_note": str(p.get("location_note", "GENERAL")).strip(),
        "confidence": str(p.get("confidence", "LOW")).strip(),    
    }


def enforce_exact_count(providers: list, desired: int) -> list:
    if not isinstance(providers, list):
        providers = []
    normalized = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        entry = normalize_provider(p)
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
        mloc = re.search(r"\b(?:in|at|near|around)\s+([A-Za-z0-9 .,\-']{2,60})", (query or ""), flags=re.I)
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

    used_frontend = False

    loc_from_ai = ai_data.get("location")
    if isinstance(loc_from_ai, str) and re.search(r'\b(near|nearby|here|around)\b', loc_from_ai.lower()):
        wants_frontend = True
    else:
        wants_frontend = requests_frontend_location(query)

    if wants_frontend and frontend_location:
    # User said "near me" and we have frontend location
        ai_data["location"] = frontend_location
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
        used_frontend = True
    elif ai_data.get("location"):
    # Location explicitly given in user input
        parsed["state"] = "complete" if ai_data.get("service") else "need_service"
    else:
    # No location info at all
        parsed["state"] = "need_location"
        ai_data["location"] = None


    with conversation_lock:
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                try:
                    past = json.loads(msg["content"])
                    past_ai = past.get("ai_data", {})
                    if not ai_data.get("service") and past_ai.get("service"):
                        ai_data["service"] = past_ai.get("service")
                    if not ai_data.get("location") and past_ai.get("location") and not used_frontend:
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
            "Important: Only include providers physically located in the specified location. "
            "Do NOT include providers from Pakistan unless the user explicitly asked for Pakistan.\n\n"
            "For each provider include keys: name, phone, details, address, location_note (EXACT|GENERAL), confidence (HIGH|MEDIUM|LOW).\n"
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

        providers = [normalize_provider(p) for p in new_data.get("providers", [])]
        missing_source = any(not p.get("source") for p in providers)
        if missing_source:
            result = {
                "valid": False,
                "message": "Provider results lacked source URLs for one or more entries — refusing to return unverified providers.",
                "state": "error",
                "providers": [],
                "suggestions": [],
                "ai_data": ai_data,
                "usage_report": {
                    "error": "missing_sources_in_providers",
                    "raw_providers_sample": providers[:3],
                    "timestamp": datetime.now().isoformat()
                }
            }
            try:
                with conversation_lock:
                    conversation.append({"role": "assistant", "content": json.dumps(result)})
            except Exception:
                pass
            return result

        parsed_providers = enforce_exact_count(providers, desired)
        parsed["providers"] = parsed_providers
        parsed["state"] = "complete"
    else:
        parsed.setdefault("providers", [])
        parsed["state"] = parsed.get("state", "error")

    result = {
        "valid": True,
        "message": parsed.get("message", f"Here are {ai_data.get('count')} providers.") if isinstance(parsed, dict) else "",
        "state": parsed.get("state", "error"),
        "providers": parsed.get("providers", []),
        "suggestions": parsed.get("suggestions", []),
        "ai_data": parsed.get("ai_data", {"intent": None, "service": None, "location": None, "confidence": 0.0}),
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
