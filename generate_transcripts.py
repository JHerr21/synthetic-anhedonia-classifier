from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq, RateLimitError, APIConnectionError, InternalServerError
import re
import argparse
import json
import random
import time

MODEL = "openai/gpt-oss-120b"

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY") 
)

SYSTEM_PROMPT = """
ROLE
You generate realistic, de-identified synthetic transcripts of clinical
interviews for a research dataset. Each transcript belongs to one of two arms:
HIGH anhedonia or LOW anhedonia. The two arms must be matched on everything
except hedonic capacity. A reader should not be able to tell the arm from the
interviewer's questions, the transcript length, the patient's diagnosis, or the
amount of ordinary disfluency, only from how the patient talks about pleasure,
interest, and anticipation.

PARAMETERS (given in the user message)
- Anhedonia arm: high | low
- Underlying context: major depressive disorder | schizophrenia, negative
  symptoms | Parkinson's disease
- Anhedonia subtype emphasis (high arm only): anticipatory | consummatory |
  social | mixed
- Severity (high arm only): moderate | severe
- Patient profile: age, sex, life context
- Number of exchanges
- Include risk screen: yes | no

FIXED INTERVIEW PROTOCOL (identical in both arms)
The interviewer covers these topics in this order, phrased naturally. It may
add one brief follow-up per topic in either arm. Follow-ups must not be more
frequent or more probing in one arm than the other.
1. How have things been going lately?
2. Walk me through a typical day this past week.
3. What do you do in your free time these days?
4. Tell me about the last time you did something you enjoyed.
5. How was that compared to how it would have felt in the past?
6. Who do you spend time with? How is that for you?
7. How has your appetite been? Do you enjoy your food?
8. How has your sleep been?
9. Is there anything coming up that you're looking forward to?
10. When something good happens, how do you usually react?
11. (Context-relevant question about the underlying condition, e.g.,
    mood, medication, or physical symptoms.)
12. Is there anything else you think I should know?
If the risk screen is enabled, add a standard brief risk screen after topic 11
in both arms.

SHARED BASELINE (both arms)
- Ordinary speech disfluency is present in everyone: occasional "um," "uh,"
  restarts, and brief [pause] markers at a normal conversational rate.
- Every patient gives some short answers and some longer ones.
- Context overlays apply in both arms, independent of anhedonia:
  Parkinson's: occasional [quietly], slowed responses, frustration about motor
  limits. Schizophrenia: some concreteness or mild tangentiality. MDD: low mood,
  worry, guilt, poor sleep or concentration.
- The patient never uses clinical terms ("anhedonia," "flat affect").

HIGH ANHEDONIA ARM: patient speech markers
Express only through how the patient talks:
1. Elaboration drops specifically on pleasure, interest, and social topics
   (topics 3–7, 9–10). Moderate: brief, needs prompting. Severe: frequent
   one-word or "I don't know" answers on these topics.
2. Longer hesitation before answering pleasure-related questions ([long pause]
   more frequent than baseline).
3. Few positive-emotion words; enjoyment is described flatly ("it was fine").
4. Enjoyment is framed mostly in the past tense ("I used to...").
5. Anticipatory emphasis: struggles to name anything to look forward to.
   Consummatory emphasis: still does activities but they "don't feel like
   anything." Social emphasis: withdraws from people without distress, little
   interest in connection.
6. Rarely volunteers detail or asks questions back.
7. Allow 1–2 moments of slightly greater engagement for realism.
Anhedonia means absence of pleasure, not dramatic sadness.

LOW ANHEDONIA ARM: patient speech markers
Hedonic capacity is intact. The patient may still have real problems from
their underlying condition. Do not make them cheerful, unusually talkative, or
upbeat. They are ordinary people who still enjoy things.
1. On pleasure topics, gives specific, concrete details (what they did, with
   whom, a small moment they liked).
2. Uses positive-emotion words at a normal rate, without exaggeration.
3. Describes enjoyment in the present tense; reduced activity is explained by
   practical limits (pain, tremor, time, money, fatigue), not lost interest.
   The key contrast: "I can't do it as much" rather than "it doesn't do
   anything for me."
4. Can name at least one specific thing they're looking forward to.
5. Shows interest in at least one relationship, even if strained.
6. For MDD: may report sadness, stress, or worry, but mood still lifts in
   response to good events.

LEAKAGE CONTROLS
- Transcript length follows the requested number of exchanges in both arms.
- Do not mention the arm, the word "anhedonia," or any label anywhere.
- Vary hobbies, jobs, and life details; do not reuse stock examples.

OUTPUT FORMAT
- Plain transcript only: no annotations, tags, commentary, or summary.
- First line, identical in both arms: [Synthetic transcript, simulated
  clinical interview]
- Speaker labels: "Interviewer:" and "Patient:"

Before writing, silently plan how the assigned arm's markers appear across the
protocol topics, then output only the final transcript.
"""

DEFAULT_FORMAT = (
    "Semi-structured clinical interview covering daily activities, hobbies, "
    "relationships, appetite/sleep, and future plans; interviewer style modeled "
    "on items from instruments such as the SHAPS, MADRS item 8, or CAINS."
)

CONTEXTS = [
    "major depressive disorder",
    "schizophrenia, negative symptoms",
    "Parkinson's disease",
]

SUBTYPES = ["anticipatory", "consummatory", "social", "mixed"]
SEVERITIES = ["moderate", "severe"]

def random_profile(context: str, rng: random.Random) -> dict:
    age_ranges = {
        "major depressive disorder": (19, 75),
        "schizophrenia, negative symptoms": (20, 55),
        "Parkinson's disease": (55, 82),
    }
    lo, hi = age_ranges.get(context, (20, 75))
    return {
        "age": rng.randint(lo, hi),
    }
    


def build_user_message(params: dict) -> str:
    lines = [
        "Generate one transcript with these parameters:",
        f"- Anhedonia arm: {params['arm']}",
        f"- Underlying context: {params['context']}",
    ]
    if params["arm"] == "high":
        lines += [f"- Anhedonia subtype emphasis: {params['subtype']}",
                  f"- Severity: {params['severity']}"]
    lines += [
        f"- Patient profile: {params['profile']['age']}-year-old",
        f"- Number of exchanges: {params['n_exchanges']}",
        f"- Include risk screen: {'yes' if params['include_risk'] else 'no'}",
    ]
    return "\n".join(lines)

def generate(client: Groq, params: dict, temperature: float, seed, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=8000,
                temperature=temperature,
                seed=seed,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(params)},
                ],
            )
            return response.choices[0].message.content
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            wait = 2 ** attempt * 5
            print(f"  API error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Generation failed after retries")

POSITIVE_WORDS = {
    "love", "loved", "enjoy", "enjoyed", "fun", "happy", "great", "excited",
    "exciting", "wonderful", "amazing", "glad", "awesome", "fantastic", "joy",
}


def marker_metrics(transcript: str) -> dict:
    """Quick sanity checks that the markers landed. Not a validated measure."""
    patient_turns = [
        line.split(":", 1)[1].strip()
        for line in transcript.splitlines()
        if line.startswith("Patient:")
    ]
    words = [w.lower().strip(".,!?\"'") for t in patient_turns for w in t.split()]
    n_words = len(words) or 1
    return {
        "patient_turns": len(patient_turns),
        "words_per_patient_turn": round(len(words) / max(len(patient_turns), 1), 1),
        "pause_markers": len(re.findall(r"\[(?:long )?pause\]", transcript)),
        "fillers": sum(w in {"um", "uh"} for w in words),
        "positive_word_rate": round(sum(w in POSITIVE_WORDS for w in words) / n_words, 4),
        "used_to_count": transcript.lower().count("used to"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-combo", type=int, default=1,
                        help="Transcripts per context x subtype x severity combination")
    parser.add_argument("--n-exchanges", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-risk", action="store_true")
    parser.add_argument("--out", default="data/complete_transcripts.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    client = Groq()  # reads GROQ_API_KEY

    rng = random.Random(args.seed)

    # Balanced design: 10 high + 10 low per context = 60 total
    design = []
    for context in CONTEXTS:
        for arm in ["high", "low"]:
            for k in range(10):
                design.append({
                    "arm": arm,
                    "context": context,
                    "subtype": SUBTYPES[k % 4] if arm == "high" else None,
                    "severity": ["moderate", "severe"][k % 2] if arm == "high" else None,
                    "profile": random_profile(context, rng),
                    "n_exchanges": args.n_exchanges,
                    "include_risk": args.include_risk,
                })
    rng.shuffle(design)  # interleave arms so API drift isn't confounded with label

    total = len(design)
    print(f"Generating {total} transcripts -> {args.out}")

    with open(args.out, "w", encoding="utf-8") as f:
        for i, params in enumerate(design, start=1):
            print(f"[{i}/{total}] {params['arm']} | {params['context']} | "
                  f"{params['subtype']} | {params['severity']}")
            transcript = generate(client, params, args.temperature,
                                  seed=args.seed + i)
            record = {
                "id": i,
                "label": 1 if params["arm"] == "high" else 0,
                "params": params,
                "model": MODEL,
                "temperature": args.temperature,
                "seed": args.seed + i,
                "transcript": transcript,
                "metrics": marker_metrics(transcript),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()