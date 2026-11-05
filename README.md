# wazoo-voice-agent

A voice agent that answers from a Wazoo world: AssemblyAI streaming STT, a
retrieval pre-pass against the world while you are still speaking, then a
streamed DeepSeek answer spoken through AssemblyAI streaming TTS. If keyword
retrieval finds nothing, it falls back to a tool loop with `@wazoo/tools`.

It holds one TTS session across turns and keeps the last three exchanges in
context, so follow-ups like "And what does he manage?" resolve the pronoun
against its own previous answer.

## Run

```sh
npm install
export ASSEMBLYAI_API_KEY=...   # also used for STT and TTS
export DEEPSEEK_API_KEY=...     # the answering model
export WORLDS_TOKEN=wzw_...     # world token (mint from api.wazoo.dev)
export WORLD_ID=w_...           # the world to recall from

node voice-agent.mjs question.pcm [question2.pcm ...]
```

Each input file is one turn of raw 16 kHz s16le mono PCM, in order. The agent
writes the reply to `answer.pcm` (24 kHz s16le mono). To listen:

```sh
ffmpeg -f s16le -ar 24000 -ac 1 -i answer.pcm answer.wav
```

With a mic, feed the capture stream through the same path (see
[the guide](https://docs.wazoo.dev/guides/voice-agent) for the browser and
mic variants).

## How it stays fast

The pipeline is retrieve-then-speak rather than answer-then-speak:

- While the user is still talking, partial STT transcripts of four or more
  words already kick off keyword retrieval in parallel.
- Search runs one query per meaningful token (three max, stopwords stripped)
  and merges subjects, then one batched SPARQL query fetches all hits.
- Answers stream, and complete sentences start playing before the stream
  ends. The tool loop only runs when retrieval came up empty.
- If a TTS session drops mid-reply, it reconnects and keeps speaking.

On a hosted world, the first word of the answer lands roughly four seconds
after the question ends; follow-up turns reuse prefetch and land sooner.

## Notes

- Keyword-only recall matches literal substrings, so pronoun-only follow-ups
  can miss and fall back to the slower tool loop. Swapping `retrieve()` for a
  hybrid (embeddings) query fixes that once your world has embeddings enabled.
- The `answerStream` prompt keeps answers grounded in the fetched facts and
  refuses to name things the facts do not state.
