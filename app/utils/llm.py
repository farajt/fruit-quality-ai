# ============================================================
# llm.py — LLM enrichment with forced complete responses
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv(override=True)


def build_prompt(result: dict) -> str:
    fruit     = result["fruit"].capitalize()
    cond      = result["condition"].capitalize()
    score     = result["freshness_score"]
    grade     = result["grade"]
    label     = result["grade_label"]
    conf      = result["confidence"]
    shelf     = result["shelf_life"]
    risk      = result["risk_level"]
    rec       = result["recommendation"]

    # Tailored instructions based on condition
    if cond == "Fresh":
        consumption_hint = (
            f"When should the user consume this {fruit.lower()}? "
            f"It has {shelf} shelf life remaining. Give specific day ranges.")
        nutritional_hint = (
            f"What specific nutrients are at their peak in a fresh "
            f"{fruit.lower()} with freshness score {score:.2f}? "
            f"Mention vitamins, minerals, antioxidants with approximate values.")
    else:
        consumption_hint = (
            f"This {fruit.lower()} is spoiled. Advise the user clearly "
            f"NOT to consume it and explain the health risks.")
        nutritional_hint = (
            f"How does a freshness score of {score:.4f} (near 0 = fully spoiled) "
            f"affect the nutritional value of a {fruit.lower()}? "
            f"Which nutrients degrade first during spoilage?")

    return f"""You are a food science expert and nutritionist.
A fruit quality AI analyzed an image and produced the measurements below.
Write a detailed report using ONLY the provided data. Do NOT invent numbers.
ALL FOUR sections are REQUIRED — never leave any section blank.

ANALYSIS DATA:
- Fruit type: {fruit}
- Condition: {cond}
- Freshness score: {score:.4f} (scale 0.0 = fully spoiled → 1.0 = perfectly fresh)
- Quality grade: {grade} ({label})
- AI confidence: {conf:.2%}
- Estimated shelf life: {shelf}
- Risk level: {risk}
- Recommendation: {rec}

INSTRUCTIONS:
Write EXACTLY these four sections. Each section: 2-3 sentences, plain English for consumers.
Do NOT skip or abbreviate any section. Do NOT say "Not applicable".

QUALITY SUMMARY:
[Describe current state of this {fruit.lower()} based on the freshness score {score:.4f} and grade {grade}. What does this score mean for the fruit physically?]

STORAGE ADVICE:
[Give specific temperature (°C and °F), humidity %, and location recommendations for a {cond.lower()} {fruit.lower()} with this exact freshness score. Should it be refrigerated?]

CONSUMPTION WINDOW:
[{consumption_hint}]

NUTRITIONAL IMPACT:
[{nutritional_hint}]
"""


def parse_response(text: str) -> dict:
    sections = {
        "quality_summary":    "",
        "storage_advice":     "",
        "consumption_window": "",
        "nutritional_impact": "",
    }
    keys = {
        "QUALITY SUMMARY":    "quality_summary",
        "STORAGE ADVICE":     "storage_advice",
        "CONSUMPTION WINDOW": "consumption_window",
        "NUTRITIONAL IMPACT": "nutritional_impact",
    }
    current, buf = None, []
    for line in text.split("\n"):
        ls, matched = line.strip(), False
        for header, key in keys.items():
            if header in ls.upper():
                if current and buf:
                    sections[current] = " ".join(
                        b for b in buf if b).strip()
                current, buf, matched = key, [], True
                break
        if not matched and current:
            c = ls.strip(":-•*[]")
            if c:
                buf.append(c)
    if current and buf:
        sections[current] = " ".join(
            b for b in buf if b).strip()
    return sections


def get_llm_report(result: dict) -> dict:
    load_dotenv(override=True)
    groq_key   = os.getenv("GROQ_API_KEY",   "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not groq_key and not gemini_key:
        return {"status": "unavailable",
                "message": "No API keys found in .env file."}

    prompt     = build_prompt(result)
    text       = None
    model_used = "unavailable"

    # ── Try Groq ──────────────────────────────────────────
    if groq_key:
        try:
            from groq import Groq
            client   = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a food science expert. "
                            "Always complete ALL four sections fully. "
                            "Never leave any section empty or say "
                            "'Not applicable'."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.3,
            )
            text       = response.choices[0].message.content
            model_used = "Groq — Llama 3.3 70B"
        except Exception as e:
            print(f"Groq failed: {e}")

    # ── Fallback: Gemini 1.5 Flash ────────────────────────
    if text is None and gemini_key:
        for model_name, label in [
            ("gemini-1.5-flash",      "Gemini 1.5 Flash"),
            ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
        ]:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                gm = genai.GenerativeModel(model_name)
                r  = gm.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=700,
                        temperature=0.3,
                    ),
                )
                text       = r.text
                model_used = label
                break
            except Exception as e:
                print(f"{label} failed: {e}")

    if text is None:
        return {"status": "unavailable",
                "message": "Both Groq and Gemini failed."}

    parsed = parse_response(text)

    # Validate — if any section is still empty, use safe defaults
    fruit = result["fruit"].capitalize()
    cond  = result["condition"].lower()
    if not parsed["consumption_window"]:
        if cond == "fresh":
            parsed["consumption_window"] = (
                f"Consume this {fruit.lower()} within "
                f"{result['shelf_life']} for best quality and flavor.")
        else:
            parsed["consumption_window"] = (
                f"Do not consume this {fruit.lower()}. "
                f"It has exceeded its safe consumption window and "
                f"should be discarded immediately.")
    if not parsed["nutritional_impact"]:
        if cond == "fresh":
            parsed["nutritional_impact"] = (
                f"At freshness score {result['freshness_score']:.2f}, "
                f"this {fruit.lower()} retains its full nutritional profile "
                f"including vitamins, minerals, and antioxidants at peak levels.")
        else:
            parsed["nutritional_impact"] = (
                f"With a freshness score of {result['freshness_score']:.4f}, "
                f"this {fruit.lower()} has undergone significant nutrient "
                f"degradation. Vitamins C and B-complex are severely depleted "
                f"and bacterial toxins may be present.")

    return {
        "status": "success",
        "model":  model_used,
        **parsed
    }