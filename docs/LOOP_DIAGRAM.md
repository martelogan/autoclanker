# Loop Diagram

This is the compact loop view of `autoclanker`: ideas nudge candidate choice,
the existing eval harness runs the candidate, and the resulting observations
feed the next choice.

```mermaid
flowchart TB
  classDef start fill:#FFF3D6,stroke:#D97706,color:#6B3F00,stroke-width:1.5px;
  classDef core fill:#EEF3FF,stroke:#3550A3,color:#162853,stroke-width:1.5px;
  classDef loop fill:#EAF7EF,stroke:#2F855A,color:#163924,stroke-width:1.5px;

  subgraph guide["① Guidance"]
    ideas["💡 Rough ideas"]:::start
    surface["🗺️ Surface / options"]:::start
  end

  subgraph engine["② Candidate loop"]
    choose["🎯 Pick next candidate"]:::core
    refine["📘 Refine search"]:::core
  end

  subgraph existing["③ Existing eval harness"]
    eval["🧪 Run candidate"]:::loop
    result["📚 Record result"]:::loop
  end

  ideas -->|"steer"| choose
  surface -->|"available moves"| choose
  choose --> eval
  eval --> result
  result --> refine
  refine --> choose
```

Editable Mermaid source lives at
[`docs/assets/autoclanker_loop.mmd`](./assets/autoclanker_loop.mmd).
