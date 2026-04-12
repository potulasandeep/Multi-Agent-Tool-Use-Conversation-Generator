# Failed prompt iterations

A running log of prompts that didn't work and what we learned.
One of these will end up in DESIGN.md per the assignment requirement.

## Planner v0 — "let the LLM decide clarification"

### The failed prompt (system message)

> You are a conversation scenario designer. Given a list of tools, produce
> a user scenario. Decide whether the user should be vague at the start
> (requiring clarification) or specific (not requiring clarification),
> based on what feels natural for the scenario.

### What went wrong

Letting the LLM choose whether to force clarification broke two things at
once:

1. **Clarification rate was uncontrollable.** The LLM decided "natural"
   for about 85% of chains meant no clarification. The assignment wants
   multi-turn disambiguation as a meaningful fraction of the corpus —
   we couldn't hit any specific rate without changing the prompt.

2. **Reproducibility suffered.** The same chain could get a different
   clarification decision on different runs because the temperature-0
   determinism of the Planner doesn't extend across prompt wording
   tweaks. The diversity experiment needs Run A and Run B to be
   bit-identical at the chain level; we lost that.

### The fix

Pull the decision out of the LLM entirely. The orchestrator flips a
seeded coin (30% clarification) and passes `force_clarification` in as
a boolean. The Planner's prompt has two explicit branches — one for
each value — and the LLM only decides the **creative** content (which
parameters to withhold, how to phrase the clarification question).

### Lesson

When a behavior needs to be precisely controllable **and** reproducible,
don't delegate it to the LLM. Put the decision in Python, let the LLM
fill in the creative details.

## Assistant v0 — "here are some values you might find useful"

### The failed prompt (grounding block)

> You have access to some values from earlier tool calls. Feel free to
> use them if they are relevant to the current tool call:
>
> hotel_id: ["hot_2824", "hot_5506"]
> city: ["Paris"]

### What went wrong

This wording produced hallucinated IDs in roughly 40% of tool calls on
a test chain, even with strong models. The permissive "feel free"
framing apparently reads to the model as "these are optional
suggestions" — and when the model has seen a billion API examples in
training with invented IDs, it reverts to the distribution.

### The fix

Reframe as a hard requirement with explicit failure language:

> Available values from previous tool calls:
>   hotels.id: ["hot_2824", "hot_5506"]
>
> You MUST use values from this list whenever a parameter of the tool
> you are calling matches a key in the list. Inventing identifiers is
> a failure.

Combined with the same rule in the system prompt ("Inventing
identifiers is a failure"), hallucination dropped to roughly 5% on the
same chain.

### Lesson

LLMs treat polite suggestions as optional. When a behavior is
non-negotiable, the prompt has to say so in words that don't leave
room for interpretation. "You MUST" and "is a failure" are blunt but
they work. Save politeness for parts of the prompt where the model
genuinely has latitude.