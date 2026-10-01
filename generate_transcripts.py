import os
from groq import Groq
import random
import time

MODEL = "llama-3.3-70b-versatile"

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")  # Best practice: use os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
ROLE
You generate realistic, de-identified synthetic transcripts of clinical interviews
for simulation purposes. The patient presents with anhedonia. The transcript must
read like a real verbatim clinical transcript, not a dramatization.

PARAMETERS
Each request provides these parameters in the user message:
- Underlying context (e.g., major depressive disorder | schizophrenia, negative
  symptoms | Parkinson's disease)
- Anhedonia subtype emphasis (anticipatory | consummatory | social | mixed)
- Severity (mild | moderate | severe)
- Interview format
- Patient profile (age, sex, life context)
- Approximate number of interviewer-patient exchanges
- Whether to include a risk screen

PATIENT SPEECH SPECIFICATION
Express these ONLY through how the patient talks. The patient never names or
describes these features.
1. Verbal output: responses are short and low in elaboration. Scale by severity.
   Mild: mostly full answers with occasional minimal replies. Moderate: frequent
   1-2 sentence answers, needs follow-up prompts. Severe: many one-word or
   "I don't know" answers.
2. Response latency: mark hesitation using transcript conventions: "um," "uh,"
   [pause], [long pause]. Frequency increases with severity.
3. Emotional vocabulary: very few positive-emotion words. When pleasure is
   discussed, use flat or neutral descriptors ("it was fine," "okay I guess").
   Avoid dramatic sadness; anhedonia is absence of pleasure, not overt despair.
4. Temporal framing of enjoyment: pleasure is referenced mostly in the past
   tense ("I used to really like...").
5. Anticipatory subtype: little expectation that future events will be
   enjoyable; struggles to name things to look forward to.
6. Consummatory subtype: still does activities but reports they "don't feel
   like anything."
7. Social subtype: describes withdrawing from people without distress about it,
   with little interest in connection.
8. Engagement: rarely asks questions back, rarely volunteers new topics, gives
   vague, generalized answers when asked for specifics.
9. Context overlay: shape the remaining speech to fit the underlying context.
   Schizophrenia: mild alogia and blunted affect cues. Parkinson's: occasional
   reduced volume noted as [quietly] and slowed responses. MDD: elevated
   first-person focus and some self-critical statements.

REALISM CONSTRAINTS
- Not every answer shows every marker. Include natural variability, including
  1-2 moments of slightly greater engagement (e.g., a brief flicker of interest
  when a specific memory comes up).
- The patient uses everyday language, never clinical terms ("anhedonia," "flat
  affect," "lack of motivation" as a label).
- The interviewer behaves like a trained clinician: open questions, gentle
  follow-ups, reflective statements, no leading questions, no diagnosis stated
  during the interview.
- Avoid stereotypes and melodrama.
- Include suicidality only if the request says to include a risk screen, in
  which case the interviewer conducts an appropriate screen.
- Use realistic, non-identifying details only. Vary hobbies, jobs, and life
  details; do not default to common choices.

OUTPUT FORMAT
- Plain transcript only: no annotations, tags, commentary, or summary.
- Speaker labels: "Interviewer:" and "Patient:"
- First line: [Synthetic transcript, simulated clinical interview]

Before writing, silently plan which markers appear in which exchanges and how
severity shapes them. Output only the final transcript.
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
SEVERITIES = ["mild", "moderate", "severe"]

def random_profile(context: str) -> dict:
    # Rough age ranges by context so profiles stay plausible
    age_ranges = {
        "major depressive disorder": (19, 75),
        "schizophrenia, negative symptoms": (20, 55),
        "Parkinson's disease": (55, 82),
    }
    lo, hi = age_ranges.get(context, (20, 75))
    return {
        "age": random.randint(lo, hi),
    }


def build_user_message(params: dict) -> str:
    p = params["profile"]
    return (
        "Generate one transcript with these parameters:\n"
        f"- Underlying context: {params['context']}\n"
        f"- Anhedonia subtype emphasis: {params['subtype']}\n"
        f"- Severity: {params['severity']}\n"
        f"- Interview format: {params['format']}\n"
        f"- Patient profile: {p['age']}-year-old\n"
        f"- Length: approximately {params['n_exchanges']} exchanges\n"
        f"- Include risk screen: {'yes' if params['include_risk'] else 'no'}"
    )

def generate(client: Groq, params: dict, temperature: float,
             max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4000,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(params)},
                ],
            )
            return response.choices[0].message.content
        except (Groq.RateLimitError, Groq.APIStatusError) as e:
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
    parser.add_argument("--out", default="transcripts.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    combos = list(itertools.product(CONTEXTS, SUBTYPES, SEVERITIES))
    total = len(combos) * args.n_per_combo
    print(f"Generating {total} transcripts -> {args.out}")

    with open(args.out, "a", encoding="utf-8") as f:
        i = 0
        for context, subtype, severity in combos:
            for _ in range(args.n_per_combo):
                i += 1
                params = {
                    "context": context,
                    "subtype": subtype,
                    "severity": severity,
                    "format": DEFAULT_FORMAT,
                    "profile": random_profile(context),
                    "n_exchanges": args.n_exchanges,
                    "include_risk": args.include_risk,
                }
                print(f"[{i}/{total}] {context} | {subtype} | {severity}")
                transcript = generate(client, params, args.temperature)
                record = {
                    "params": params,
                    "model": MODEL,
                    "temperature": args.temperature,
                    "transcript": transcript,
                    "metrics": marker_metrics(transcript),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()