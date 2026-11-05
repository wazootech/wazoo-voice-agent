// Seed the demo world with a small project graph and build its search index.
// Idempotent: fixed IRIs, so re-running writes the same triples.
// Usage: platform secrets run -- node scripts/seed.mjs   (or export the env vars yourself)
// Requires WORLDS_TOKEN and WORLD_ID.
import { createClient, importWorld, reindexWorld } from "@worlds/client";

const client = createClient({
  baseUrl: "https://data.wazoo.dev",
  auth: process.env.WORLDS_TOKEN,
});
const worldId = process.env.WORLD_ID;

const turtle = `@prefix demo: <https://demo.wazoo.dev/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

demo:ethan a demo:Person ;
  rdfs:label "Ethan Davidson" ;
  demo:role "Founder" .

demo:wazoo a demo:Project ;
  rdfs:label "Wazoo" ;
  rdfs:description "World models as a service for AI agents" ;
  demo:managedBy demo:ethan .

demo:worlds a demo:Project ;
  rdfs:label "Worlds" ;
  rdfs:description "The RDF data plane: worlds store facts, hybrid search, and SPARQL" ;
  demo:managedBy demo:ethan .

demo:docs a demo:Project ;
  rdfs:label "docs.wazoo.dev" ;
  rdfs:description "The documentation site for Wazoo" ;
  demo:managedBy demo:ethan .

demo:sparql-engine a demo:Project ;
  rdfs:label "SPARQL Engine" ;
  rdfs:description "A zero-dependency SPARQL 1.1 and 1.2 engine" ;
  demo:managedBy demo:ethan .

demo:memorybench a demo:Project ;
  rdfs:label "Wazoo Memorybench" ;
  rdfs:description "A benchmark for agent memory systems" ;
  demo:managedBy demo:ethan .
`;

const imported = await importWorld({
  client,
  path: { worldId },
  body: { data: turtle, contentType: "text/turtle" },
});
console.log(`imported quads: ${imported.data?.imported?.quads ?? imported.data?.quads ?? "?"}`);

// Keyword search reads the chunk index; it stays empty until reindex runs.
await reindexWorld({ client, path: { worldId } });
console.log("reindexed");
