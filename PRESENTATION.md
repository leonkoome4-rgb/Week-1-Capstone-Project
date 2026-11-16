# AfyaPlus Triage Engine: 5-Minute Briefing

**Audience:** AfyaPlus leadership and non-technical stakeholders
**Format:** 7 slides, about 40 seconds each. Each slide has its on-screen text and a speaker script.

---

## Slide 1: The problem (0:00–0:45)

**On screen**
- Patients write to us the way they talk: long, informal, sometimes in Swahili or Sheng.
- Our backend systems need short, predictable, structured data.
- Today's AI attempts fail in three ways: chatty filler, invented medical facts, and freezing when the network drops.

**Say**
> "Every day, patients message AfyaPlus describing how they feel, in their own words. Some messages are routine. A few are emergencies, like a pregnant mother with warning signs of preeclampsia. Our systems can't read free text, so we need something that turns a message into a clear decision: is this an emergency, and where should it go? Earlier AI attempts chatted instead of answering, sometimes made up medical facts, and stopped working when the internet was slow. None of that is acceptable in healthcare."

---

## Slide 2: What we built (0:45–1:25)

**On screen**
A patient message goes through four steps:
1. The AI reads it.
2. We check that the answer is in the right format.
3. A safety check runs.
4. The message is routed.

The output is always the same four fields:
- Emergency? Yes/No
- Symptoms found
- Short reasoning
- Where to send the patient

**Say**
> "We built a triage engine. It reads the patient's message and always answers in the same four fields, which our systems can use straight away. If the AI's answer is in the wrong shape, we reject it automatically. The patient is never left without a route."

---

## Slide 3: Live example (1:25–2:05)

**On screen** (from `samples/02_chest_pain_cloud_timeout_fallback.log`)
> *"My father is 58 and he has crushing chest pain spreading to his left arm…"*
> → **CRITICAL EMERGENCY → EMERGENCY_SERVICES**

> *"I'm 20 weeks pregnant and feeling a bit more tired than usual…"*
> → **Non-critical → MATERNAL_HEALTH_TEAM**

**Say**
> "Here are two real runs. A possible heart attack goes straight to emergency services. A routine pregnancy question goes to the maternal health team, so it doesn't use up emergency capacity. It also works in Swahili: a patient vomiting blood was correctly flagged as an emergency."

---

## Slide 4: Why these models (2:05–2:50)

**On screen**

| | Cloud: GPT-4o-mini | Local: Llama 3.2 (3B) |
|---|---|---|
| Speed | Typically 1–3 s (re-measure with your API key) | About 7–9 s on a laptop (measured) |
| Cost | Fractions of a cent per message | Free after hardware |
| Works offline | No | Yes |
| Patient data leaves the clinic | Yes | No |

**Say**
> "We use two AI engines. The main one is GPT-4o-mini, a small, fast and inexpensive cloud model. It's accurate enough for sorting messages and costs very little. If the cloud is slow or unreachable for more than four seconds, the system switches automatically to a model running on our own computer. That backup is slower, but it works with no internet and keeps patient data on site. That matters for clinics with unreliable connectivity."

---

## Slide 5: Built-in safety (2:50–3:35)

**On screen**
- The AI is told exactly what it is and isn't: no diagnosing, no prescribing, no small talk.
- It must reason step by step: symptoms, then warning signs, then a decision.
- A rule-based safety net catches known red flags even if the AI misses them. It can only raise urgency, never lower it.
- If every AI engine is down, the message goes to a human nurse. Nothing is dropped.
- Attempts to trick the AI ("ignore your instructions…") are ignored.

**Say**
> "Safety is designed in at several layers. The AI follows a strict job description and must show its reasoning. Separately, a simple rule list catches things like 'pregnant plus blurred vision' or 'vomiting blood'. If the AI ever under-calls one of these, the rules overrule it. They can only make a case more urgent, never less. If everything fails, the message goes to a nurse."

---

## Slide 6: Risks and limitations (3:35–4:25)

**On screen**
- **Not a medical device.** This is a sorting aid; clinicians make the decisions.
- **Missed emergencies.** Unusual wording can slip past both the AI and the rules. A nurse must still review a sample.
- **Over-triage.** Being cautious sends some routine cases to urgent queues, which costs capacity.
- **Local model is weaker and slower.** It's a backup, not the main engine.
- **Privacy.** The cloud pathway sends message text to OpenAI, so we need a data-processing agreement and Kenya Data Protection Act compliance.
- **Not yet validated** on real AfyaPlus patient data or a broad Swahili/Sheng sample.

**Say**
> "This is a prototype, so I want to be clear about the risks. It doesn't replace a clinician. It can still miss an emergency written in an unusual way, so nurses must keep reviewing a sample. We deliberately lean towards caution, which means some routine cases get escalated. The backup model is slower and less reliable. Sending patient messages to a cloud provider has privacy and legal implications we need to settle before going live."

---

## Slide 7: Recommendation and next steps (4:25–5:00)

**On screen**
1. Pilot with nurses reviewing 100% of AI decisions (shadow mode).
2. Measure missed emergencies and over-triage on real, anonymised messages.
3. Expand Swahili/Sheng testing.
4. Agree data-protection terms before the cloud pathway touches real patient data.

**Say**
> "My recommendation is a shadow pilot. The engine runs alongside our nurses, who check every decision. We measure how often it misses an emergency and how often it over-escalates, and only then let it route live traffic. The engine is ready for that pilot today."
