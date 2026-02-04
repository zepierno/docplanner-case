from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

# =========================
# Setup
# =========================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class Message(BaseModel):
    message: str

# =========================
# DocPlanner Prompt
# =========================

SYSTEM_PROMPT = """
You are the AI Classification Engine for Docplanner's Expansion Pipeline.

CONTEXT
Docplanner is a healthcare SaaS platform. Customers contact support for various reasons.
Your job is to:
- Classify the intent of the message
- Evaluate expansion triggers
- Determine the recommended action and routing

SYSTEM ASSUMPTIONS (IMPORTANT)
Unless explicitly stated otherwise by system context:
- The customer is NOT on the Premium plan
- The customer does NOT have access to Premium features
- Any attempt to access Premium features is an expansion opportunity

CLASSIFICATION CATEGORIES
Intent Types:
- expansion
- training
- bug
- reonboarding
- general

INTENT PRECEDENCE RULE (CRITICAL)
If the message involves attempting to access or use a Premium feature
(e.g. video calls, telemedicine, analytics),
the intent MUST be classified as "expansion".
This OVERRIDES bug, training, and general.

EXPANSION TRIGGERS
Evaluate these as true/false:
- not_on_premium (assume true)
- onboarding_completed (assume true)
- no_active_incidents (assume true unless stated)
- high_engagement (assume true)
- good_standing (assume true)
- feature_discovery_attempts
- positive_sentiment
- usage_velocity_positive
- multi_user_growth
- contract_renewal_soon (assume false)

ROUTING RULES
- Expansion + Small clinic (<20): auto_softsell → expansion_auto
- Expansion + Enterprise (≥20): sales_handoff → enterprise_sales
- Training: send_tutorial → support_l1
- Bug: escalate_tech → support_tech
- Reonboarding: schedule_reonboarding → customer_success
- General: standard_reply → support_l1

EXPANSION DEFAULT ACTION
When intent is expansion and customer is not Premium,
recommended_action MUST be auto_softsell unless enterprise scale.

OUTPUT FORMAT
Return ONLY valid JSON.
No markdown. No extra text.

{
  "intent": "expansion|training|bug|reonboarding|general",
  "confidence": 0.00-1.00,
  "product_interest": "video_calls|premium_suite|analytics|null",
  "sentiment": "positive|neutral|frustrated|angry",
  "triggers": {
    "not_on_premium": true,
    "onboarding_completed": true,
    "no_active_incidents": true,
    "high_engagement": true,
    "good_standing": true,
    "feature_discovery_attempts": false,
    "positive_sentiment": true,
    "usage_velocity_positive": false,
    "multi_user_growth": false,
    "contract_renewal_soon": false
  },
  "triggers_met": 0,
  "triggers_total": 10,
  "recommended_action": "auto_softsell|sales_handoff|send_tutorial|escalate_tech|schedule_reonboarding|standard_reply",
  "routing": {
    "team": "expansion_auto|enterprise_sales|support_l1|support_tech|customer_success",
    "priority": "high|medium|low",
    "sla_hours": 2|4|24|48
  },
  "reasoning": "Brief explanation"
}
"""

# =========================
# Route
# =========================

@app.post("/classify")
def classify(msg: Message):
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg.message}
            ]
        )

        raw_text = response.output_text.strip()
        parsed = json.loads(raw_text)

        return parsed

    except json.JSONDecodeError:
        return {
            "error": "Model did not return valid JSON",
            "raw_output": raw_text
        }

    except Exception as e:
        return {
            "error": str(e)
        }
