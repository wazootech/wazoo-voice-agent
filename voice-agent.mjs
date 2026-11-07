// Wazoo voice agent: streaming STT -> LLM with Worlds recall tools -> streaming TTS.
// Run: platform secrets run -- node voice-agent.mjs <16kHz-wav> [more.wav ...]
// Each file is one conversation turn, in order.
// (mic version: same loop with a real capture source; file input lets it run
//  anywhere, which is also how the guide's smoke test works.)
import { readFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { AssemblyAI } from "assemblyai";
import { streamText, stepCountIs } from "ai";
import { deepseek } from "@ai-sdk/deepseek";
import { createRecallTools } from "@wazoo/tools";

// ---------- the world (hosted data plane) ----------------------------------
import { createClient, searchWorld, sparqlWorld } from "@worlds/client";

const worldId = process.env.WORLD_ID;
const hostedClient = createClient({
  baseUrl: "https://data.wazoo.dev",
  auth: process.env.WORLDS_TOKEN,
});

const hosted = {
  search: async ({ query, topK, minScore }) => {
    const response = await searchWorld({
      client: hostedClient,
      path: { worldId },
      body: { query, limit: topK, minScore },
    });
    return response.data;
  },
  sparql: async ({ query }) => {
    const response = await sparqlWorld({
      client: hostedClient,
      path: { worldId },
      body: { query },
    });
    return { kind: "select", data: response.data };
  },
};
const recall = createRecallTools({ client: hosted });

// retrieve-then-speak: deterministic recall pre-pass, then a tool-less
// streaming answer. The tool loop below is the fallback when retrieval
// comes up empty.
// Query-time rewrite (worlds-sdk-ts#215 H1-lite): the hosted index matches
// literal substrings, so pronouns and inflections lose recall. Substitute
// pronouns with entity labels from recent conversation, expand tokens with
// light stemming, all without embeddings.
const PRONOUNS = new Set([
  "he", "she", "it", "his", "her", "its", "him", "they", "them", "their",
]);

const GREETINGS = new Set(["hey", "hi", "hello", "ok", "okay", "so", "well"]);

function rewrite(question, history) {
  // Collect the proper names the conversation has already established.
  // Answers only: STT-mangled questions spell entities inconsistently.
  const names = new Set();
  for (const turn of history.slice(-3)) {
    for (const m of turn.answer.matchAll(
      /\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b/g,
    )) {
      const words = m[0].split(/\s+/);
      while (words.length && GREETINGS.has(words[0].toLowerCase())) {
        words.shift();
      }
      const name = words.join(" ");
      if (name && !PRONOUNS.has(name.toLowerCase())) names.add(name);
    }
  }
  let resolved = question;
  for (const name of names) {
    resolved = resolved.replace(
      new RegExp(`\\b(his|her|its|their)\\b`, "gi"),
      `${name}'s`,
    );
    resolved = resolved.replace(
      new RegExp(`\\b(he|she|it|they)\\b`, "gi"),
      name,
    );
    resolved = resolved.replace(
      new RegExp(`\\b(him|her|it|them)\\b`, "gi"),
      name,
    );
  }
  return resolved;
}

const SUFFIXES = ["'s", "s", "es", "ing", "ed"];

function expand(token) {
  const variants = new Set([token]);
  for (const suf of SUFFIXES) {
    if (token.endsWith(suf) && token.length - suf.length >= 3) {
      variants.add(token.slice(0, -suf.length));
    }
  }
  for (const suf of SUFFIXES) {
    if (!token.endsWith(suf)) variants.add(token + suf);
  }
  return [...variants];
}

async function retrieve(question, history = []) {
  const rewritten = rewrite(question, history);
  // Keyword-only search matches literal substrings, so query per keyword
  // and merge subjects. (Swap for one hybrid query once embeddings are on.)
  const base = rewritten
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t))
    .slice(0, 3);
  // Expansions come after every base token so a content word can never be
  // crowded out of the search budget by variant junk.
  const tokens = [...new Set([...base, ...base.flatMap(expand)])].slice(0, 6);
  const bySubject = new Map();
  await Promise.all(
    tokens.map(async (token) => {
      const search = await hosted.search({ query: token, topK: 5 });
      for (const h of search?.results ?? []) {
        if (
          h.subject &&
          /^https?:\/\//.test(h.subject) &&
          !bySubject.has(h.subject)
        ) {
          bySubject.set(h.subject, h);
        }
      }
    }),
  );
  const hits = [...bySubject.values()].slice(0, 6);
  const facts = [];
  if (hits.length > 0) {
    const iris = [...new Set(hits.map((h) => h.subject))];
    const values = iris.map((i) => `<${i}>`).join(" ");
    // Forward: the hit subject's own triples. Reverse: triples that point
    // AT the hit subject ("who manages Ethan" lives on the projects).
    const [fwd, rev] = await Promise.all([
      hosted.sparql({
        query: `SELECT ?s ?p ?o WHERE { VALUES ?s { ${values} } ?s ?p ?o }`,
      }),
      hosted.sparql({
        query: `SELECT ?s ?p ?o WHERE { VALUES ?o { ${values} } ?s ?p ?o }`,
      }),
    ]);
    for (const b of fwd?.data?.results?.bindings ?? []) {
      facts.push(`${b.s.value} ${b.p.value} ${b.o.value}`);
    }
    for (const b of rev?.data?.results?.bindings ?? []) {
      facts.push(`${b.s.value} ${b.p.value} ${b.o.value}`);
    }
  }
  return { hits, facts, tokens };
}
const STOPWORDS = new Set([
  "the", "and", "for", "what", "who", "how", "are", "you", "hey",
  "was", "his", "her", "its", "does", "did", "can", "have", "has",
  "with", "that", "this", "from", "about", "into", "list", "tell",
]);

function answerStream(question, factsBlock) {
  return streamText({
    model: deepseek("deepseek-v4-flash"),
    system:
      "You are a voice agent answering from a Wazoo knowledge world. " +
      "Answer ONLY from the facts below (N-Triples: subject predicate " +
      "object; rdfs:label gives display names). If the facts do not " +
      "answer the question, say what is missing. Answer in two or three " +
      "short spoken sentences. No markdown.\n\n" + factsBlock,
    prompt: question,
  });
}

// ---------- agent loop fallback --------------------------------------------
function agentStream(question) {
  return streamText({
    model: deepseek("deepseek-v4-flash"),
    system:
      "You are a voice agent answering spoken questions from a Wazoo " +
      "knowledge world. Use the tools to ground your answer: try " +
      "executeSparql for structured questions (who/what manages, lists, " +
      "relations) and searchWorld for fuzzy lookups; searchWorld matches " +
      "exact text inside literals, so use short distinctive words from " +
      "the question. Never announce a lookup, never say 'let me check'. " +
      "When tools return nothing, say you could not find it. Answer in " +
      "two or three short spoken sentences. No markdown.",
    tools: recall,
    stopWhen: stepCountIs(3),
    prompt: question,
  });
}

// sentence split for early playback
function sentences(text) {
  return text
    .replace(/\n+/g, " ")
    .match(/[^.!?]+[.!?]*/g)
    ?.map((s) => s.trim())
    .filter(Boolean) ?? [text];
}

// ---------- assemblyai legs -------------------------------------------------
const client = new AssemblyAI({ apiKey: process.env.ASSEMBLYAI_API_KEY });

async function speak(session, text, out) {
  const t = Date.now();
  let first = 0;
  // chunk destination is `out` in the caller's scope
  for await (const chunk of session.speak(text)) {
    if (!first) first = Date.now();
    out.write(chunk); // 24 kHz s16le mono PCM
  }
  return first ? first - t : 0;
}

// ---------- run one turn end to end ----------------------------------------
const clips = process.argv.slice(2);
if (clips.length === 0) {
  console.error("usage: node voice-agent.mjs <16kHz wav> [more.wav ...]");
  process.exit(1);
}

const out = createWriteStream("answer.pcm");
let firstAudio = 0;

async function freshSession() {
  const tok = await client.tts.createTemporaryToken({ expires_in_seconds: 120 });
  const sess = client.tts.session({ token: tok, voiceId: "aai-jane", language: "english" });
  await sess.connect();
  return sess;
}

// Speak one sentence, reconnecting if the session dropped.
async function speakOrReconnect(currentRef, text) {
  try {
    return await speak(currentRef.current, text, out);
  } catch {
    console.error("tts session dropped, reconnecting");
    try { await currentRef.current.close(); } catch {}
    currentRef.current = await freshSession();
    return speak(currentRef.current, text, out);
  }
}

// ---- per-turn pipeline: STT file -> recall -> streamed spoken answer ----
const history = []; // {question, answer} for conversation context

async function runTurn(wavPath, tts) {
  const t = Date.now();
  const sttToken = await client.streaming.createTemporaryToken({
    expires_in_seconds: 120,
  });
  const transcriber = client.streaming.transcriber({
    speechModel: "universal-3-6-pro",
    encoding: "pcm_s16le",
    sampleRate: 16000,
    connectTimeout: 10000,
    token: sttToken,
  });

  let prefetch = null;
  let prefetchText = "";
  const finals = [];
  transcriber.on("turn", (turn) => {
    if (turn.end_of_turn && turn.transcript) {
      finals.push(turn.transcript);
    } else if (
      !prefetch &&
      turn.transcript &&
      turn.transcript.trim().split(/\s+/).length >= 4
    ) {
      prefetchText = turn.transcript.trim();
      prefetch = retrieve(prefetchText, history).catch(() => null);
    }
  });

  await transcriber.connect();
  const pcm = await readFile(wavPath);
  const frame = 3200; // ~100ms of 16kHz s16le mono
  for (let off = 0; off < pcm.length; off += frame) {
    transcriber.sendAudio(pcm.subarray(off, off + frame));
    await new Promise((r) => setTimeout(r, 100));
  }
  await new Promise((r) => setTimeout(r, 400));
  await transcriber.close();
  if (finals.length === 0) {
    console.error("no final turn arrived");
    process.exit(1);
  }

  const finalText = finals.join(" ").trim();
  console.error(`turn heard (${Date.now() - t}ms): "${finalText}"`);

  // Recall: reuse prefetch unless the final transcript changed materially.
  // Reuse the prefetch unless the final transcript changed materially
  // (STT spelling wobble like "Wazu"/"Wazoo" should not cost a re-run).
  const rewritten = rewrite(finalText, history);
  const norm = (t) => t.toLowerCase().replace(/[^a-z0-9]/g, "");
  let retrieved = prefetch ? await prefetch : null;
  if (!retrieved || norm(prefetchText) !== norm(finalText)) {
    retrieved = await retrieve(finalText, history);
  }
  console.error(
    `retrieved (${Date.now() - t}ms): ${retrieved.facts.length} facts from ${retrieved.hits.length} hits`,
  );
  if (history.length > 0) console.error(`rewritten query: ${rewritten}`);

  // Answer stream: facts context plus recent conversation history.
  const convo = history
    .slice(-3)
    .map((h) => `User: ${h.question}\nAssistant: ${h.answer}`)
    .join("\n");
  const context =
    (convo ? "Recent conversation:\n" + convo + "\n\n" : "") +
    "Facts:\n" + retrieved.facts.join("\n") +
    "\n\nAnswer the CURRENT question. Resolve pronouns like 'he' against " +
    "the conversation; facts are for the current question.";

  let result;
  if (retrieved.facts.length > 0) {
    result = answerStream(finalText, context);
  } else {
    console.error("retrieval empty; falling back to tool loop");
    result = agentStream(finalText);
  }

  let pending = "";
  let answer = "";
  for await (const delta of result.textStream) {
    answer += delta;
    pending += delta;
    const parts = pending.match(/[^.!?]+[.!?]+\s*/g);
    if (parts) {
      const complete = parts.join("");
      pending = pending.slice(complete.length);
      for (const s of sentences(complete)) {
        const latency = await speakOrReconnect(tts, s);
        if (!firstAudio) {
          firstAudio = latency;
          console.error(`first audio (${Date.now() - t}ms)`);
        }
      }
    }
  }
  if (pending.trim()) await speakOrReconnect(tts, pending.trim());
  if (!answer.trim()) {
    const steps = await result.steps;
    const digest = JSON.stringify(
      steps.flatMap((st) => st.toolResults ?? []).map((r) => r.output),
    ).slice(0, 4000);
    console.error("no spoken text after tool rounds; summarizing");
    const followup = streamText({
      model: deepseek("deepseek-v4-flash"),
      system:
        "You are a voice agent. A tool loop gathered facts from a Wazoo " +
        "knowledge world but produced no answer text. Answer the original " +
        "question from the tool results below in two or three short spoken " +
        "sentences. No markdown.\n\nTool results: " + digest,
      prompt: finalText,
    });
    pending = "";
    for await (const delta of followup.textStream) {
      answer += delta;
      pending += delta;
      const parts = pending.match(/[^.!?]+[.!?]+\s*/g);
      if (parts) {
        const complete = parts.join("");
        pending = pending.slice(complete.length);
        for (const s of sentences(complete)) {
          const latency = await speakOrReconnect(tts, s);
          if (!firstAudio) {
            firstAudio = latency;
            console.error(`first audio (${Date.now() - t}ms)`);
          }
        }
      }
    }
    if (pending.trim()) await speakOrReconnect(tts, pending.trim());
  }
  console.error(`turn done (${Date.now() - t}ms): "${answer}"`);
  history.push({ question: finalText, answer });
}

const currentRef = { current: await freshSession() };
for (const clip of clips) {
  await runTurn(clip, currentRef);
}
try { await currentRef.current.close(); } catch {}
console.error(`done: first audio +${firstAudio}ms`);
