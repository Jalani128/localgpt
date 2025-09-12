# inference.py

from openai import OpenAI
import json
import threading
import re
from datetime import datetime
from config import OPENAI_API_KEY
import logging

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Conversation memory (system rules) ---
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

For providers: 
- Generate 5-8 providers by default (minimum 3, maximum 20)
- Use realistic Pakistani business patterns (names, +92-XXX-XXXXXXX phones, real areas)
- Ensure no duplicate business names
- Add "Please verify contact details independently" in details
- Vary business sizes (established companies, local shops, specialists)

MANDATORY: Return ONLY valid JSON in this EXACT format (no extra text):
{
  "valid": true,
  "message": "Natural conversational response",
  "state": "complete|need_service|need_location|redirect|error",
  "providers": [{"name": "Business Name", "phone": "+92-XXX-XXXXXXX", "details": "Service description. Please verify contact details independently.", "address": "Full address in Pakistan", "location_note": "EXACT|GENERAL|NEARBY", "confidence": "HIGH|MEDIUM|LOW"}],
  "suggestions": ["short", "service", "types"],
  "ai_data": {"intent": "detected_intent", "service": "service_type_or_null", "location": "location_or_null", "confidence": 0.9},
  "usage_report": {}
}"""
    }
]

# Lock for safe memory usage
conversation_lock = threading.Lock()

# Logger
logger = logging.getLogger("localgpt2.inference")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(_h)


def detect_provider_count(query: str) -> int:
    """Detect requested provider count. Default = 5, max = 20."""
    # Check for specific requests
    if "load more" in query.lower() or "more providers" in query.lower():
        return 10
    if "all" in query.lower() or "maximum" in query.lower():
        return 20
    
    # Look for numbers in query
    match = re.search(r"\b(\d+)\b", query)
    if match:
        try:
            num = int(match.group(1))
            return min(max(num, 3), 20)  # Between 3-20
        except Exception:
            return 5
    return 5  # Default to 5 providers


def safe_json_parse(text: str) -> dict:
    """Extract and parse JSON safely from text output."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        # try to extract JSON block between { }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                return {}
        return {}


def process_query(query: str) -> dict:
    """Process user query with memory + web search + JSON output."""
    global conversation

    logger.info(f"process_query received query: {query}")
    provider_count = detect_provider_count(query)

    # Trim conversation to save tokens + append user message  
    with conversation_lock:
        if len(conversation) > 7:  # Keep system message + last 6 exchanges
            conversation = [conversation[0]] + conversation[-6:]
        conversation.append({"role": "user", "content": query})

    # Call OpenAI API with better model for JSON formatting
    logger.info(f"About to call OpenAI API with provider_count: {provider_count}")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # Latest GPT-4o model with reliable JSON formatting
        messages=conversation,
        response_format={"type": "json_object"},
        max_tokens=2000,  # More tokens for detailed provider information
        temperature=0.2  # Lower temperature for more consistent, realistic output
    )
    
    logger.info(f"OpenAI API returned. Model: {response.model}")
    logger.info(f"Usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
    logger.info(f"Raw response content: {response.choices[0].message.content}")  # Full response for debugging

    # Parse output safely with detailed logging
    try:
        output_json = json.loads(response.choices[0].message.content)
        logger.info("Successfully parsed JSON response")
        logger.info(f"Parsed JSON keys: {list(output_json.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.info("Attempting fallback JSON extraction...")
        output_json = safe_json_parse(response.choices[0].message.content)
        if output_json:
            logger.info("Fallback JSON extraction successful")
        else:
            logger.error("Fallback JSON extraction failed")
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        output_json = None
        
    if not output_json:
        logger.warning("Creating fallback response due to JSON parsing failure")
        output_json = {
            "valid": False,
            "message": "I'm having trouble processing that request. Please try rephrasing.",
            "state": "error",
            "providers": [],
            "suggestions": ["plumber", "electrician"],
            "ai_data": {
                "intent": "error",
                "service": None,
                "location": None,
                "confidence": 0.0,
            },
            "usage_report": {}
        }

    # Memory: carry forward missing service/location
    with conversation_lock:
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                try:
                    past_data = json.loads(msg["content"])
                    past_service = past_data.get("ai_data", {}).get("service")
                    past_location = past_data.get("ai_data", {}).get("location")

                    if not output_json.get("ai_data", {}).get("service") and past_service:
                        output_json.setdefault("ai_data", {})["service"] = past_service
                    if not output_json.get("ai_data", {}).get("location") and past_location:
                        output_json.setdefault("ai_data", {})["location"] = past_location
                except Exception:
                    pass
                break

    # Ensure provider count
    if output_json.get("state") == "complete":
        providers = output_json.get("providers", [])
        if isinstance(providers, list) and len(providers) < provider_count:
            service = output_json["ai_data"].get("service")
            location = output_json["ai_data"].get("location")

            provider_prompt = f"""
            Generate exactly {provider_count} unique {service} providers in {location}, Pakistan.
            
            Requirements:
            - All business names must be different and realistic
            - Use Pakistani phone format: +92-XX-XXXXXXX or 0XXX-XXXXXXX
            - Include variety: established companies, local shops, specialists
            - Real-sounding addresses in {location}
            - Add "Please verify contact details independently" in details
            
            Return ONLY a JSON object with "providers" array:
            {{"providers": [{{"name": "...", "phone": "...", "details": "...", "address": "...", "location_note": "EXACT", "confidence": "HIGH"}}]}}
            """

            provider_resp = client.chat.completions.create(
                model="gpt-4o-2024-08-06",  # Same reliable model for consistency
                messages=[{"role": "user", "content": provider_prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.2
            )

            new_providers_data = safe_json_parse(provider_resp.choices[0].message.content)
            if isinstance(new_providers_data, dict) and "providers" in new_providers_data:
                new_providers = new_providers_data["providers"]
                if isinstance(new_providers, list) and len(new_providers) > 0:
                    output_json["providers"] = new_providers

    # Calculate provider count first (needed for usage report)
    providers_count = len(output_json.get("providers") or [])

    # Add usage report with current GPT-4o-2024-08-06 pricing
    try:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens

        # GPT-4o-2024-08-06 pricing (as of Sept 2025)
        # Input: $5.00 per 1M tokens = $0.005 per 1K tokens
        # Output: $15.00 per 1M tokens = $0.015 per 1K tokens
        input_cost_per_1k = 0.005  # $5.00 per 1M input tokens
        output_cost_per_1k = 0.015   # $15.00 per 1M output tokens
        cost = (input_tokens / 1000 * input_cost_per_1k) + (
            output_tokens / 1000 * output_cost_per_1k
        )

        usage_report = {
            "model": response.model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens
            },
            "cost": {
                "input_cost": round(input_tokens / 1000 * input_cost_per_1k, 6),
                "output_cost": round(output_tokens / 1000 * output_cost_per_1k, 6),
                "total_cost": round(cost, 6)
            },
            "pricing": {
                "input_per_1k": input_cost_per_1k,
                "output_per_1k": output_cost_per_1k,
                "currency": "USD"
            },
            "timestamp": datetime.now().isoformat(),
            "query": query[:50] + "..." if len(query) > 50 else query,
            "provider_count": providers_count
        }
        
        output_json["usage_report"] = usage_report
        logger.info(f"Usage analytics: {json.dumps(usage_report, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to create usage report: {e}")
        output_json.setdefault("usage_report", {})

    # Save assistant reply to memory
    try:
        with conversation_lock:
            conversation.append(
                {"role": "assistant", "content": json.dumps(output_json)}
            )
    except Exception:
        pass

    logger.info(
        f"process_query result: valid={output_json.get('valid')} "
        f"state={output_json.get('state')} providers={providers_count} "
        f"query='{query[:50]}...'"
    )

    return output_json


if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        result = process_query(query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
