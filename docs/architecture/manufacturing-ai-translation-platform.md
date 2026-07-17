# Manufacturing AI Translation Platform Architecture

## Architecture Intent

The platform is a governed manufacturing-translation mechanism, not a standalone prompt or a collection of word pairs. It combines deterministic knowledge controls, contextual AI reasoning, resilient document processing, engineering review, and continuous improvement.

```mermaid
flowchart LR
    subgraph Inputs[Engineering Inputs]
        T[Text]
        D[Documents]
        H[HMI / Images]
    end

    subgraph Platform[Manufacturing AI Translation Platform]
        E[Extract and Detect Japanese]
        C[Build Manufacturing Context]
        G[Apply Controlled Knowledge]
        A[Enterprise AI Translation]
        R[Validate and Render Reviewable Output]
        E --> C --> G --> A --> R
    end

    subgraph Knowledge[Controlled Knowledge]
        K1[Glossary]
        K2[Abbreviation Standard]
        K3[Comment Standard]
        K4[Pattern Rules]
        K5[Validation and Approval Metadata]
    end

    subgraph Operations[Operational Controls]
        O1[Job Queue]
        O2[Checkpoint Recovery]
        O3[Translation Memory]
        O4[Usage and Quality Metrics]
        O5[Session-Owned Task Boundary]
        O6[OS Trust and Proxy-Aware Connectivity]
    end

    subgraph Outputs[Engineering Outputs]
        X1[Controlled English]
        X2[Translated Files]
        X3[HMI Review Map]
    end

    T --> E
    D --> E
    H --> E
    Knowledge --> G
    Operations <--> Platform
    R --> Outputs
    Outputs --> Q[Engineering Review]
    Q --> I[Approved Knowledge Improvement]
    I --> Knowledge
```

## Design Decisions

### 1. Separate detection quality from translation quality

For HMI and image inputs, incorrect cell segmentation or OCR cannot be repaired reliably by a glossary. The architecture therefore treats region detection, text recognition, terminology control, translation, and rendering as separate quality stages.

### 2. Use deterministic controls where consistency matters

Approved terminology, protected codes, file structure, checkpoints, and output reconstruction should be deterministic. Generative reasoning is used for context-sensitive language decisions rather than for every system responsibility.

### 3. Preserve engineering reviewability

Outputs retain source-to-translation traceability. HMI results use numbered regions; document results retain structural locations; glossary matches and job status can be reviewed independently.

### 4. Design for interruption and repeat work

Large files require checkpoints, retry handling, translation memory, and job state. These controls reduce rework and make the workflow operable on imperfect networks and shared infrastructure.

### 5. Build knowledge governance into the system

Engineering corrections should not remain isolated edits. Approved corrections return to controlled knowledge through an ownership and review process, improving future translations and supporting Yokoten.

### 6. Make task ownership explicit on a shared pilot

Framework-level session state is not sufficient when background jobs are stored in a shared database. Each document job, lookup, stop action, preview, result, and checkpoint must carry the same opaque session owner. This prevents one anonymous browser session from operating on another session's work through the application UI.

The pilot uses an opaque client-session identifier to maintain continuity across refreshes. It is deliberately described as session isolation rather than authentication: a production service still requires enterprise identity, authorization, retention controls, and auditability.

### 7. Show which governed knowledge was used

Applying a controlled term internally is not enough for engineering review. When a translation uses a governed glossary entry, the review surface should expose the complete approved record available to the application, including validation and approval metadata. Repeated use of one term is consolidated into one row with an application count so engineers can understand both the decision and its reuse without navigating a separate source file.

The public repository documents this behavior but does not publish the controlled glossary, validator identities, approver identities, or production outputs.

### 8. Treat enterprise connectivity as an application dependency

API reliability depends on more than retry logic. A hosted process must use the operating system's trusted certificate store and the approved network-proxy path. Disabling certificate verification or hardcoding an infrastructure address is not an acceptable long-term reliability strategy.

The shared-host startup design resolves the approved system proxy at process start, retains certificate verification, and classifies connection failures by certificate, timeout, authentication, API status, or network path. An IT-managed service remains the production target.

## Deployment Evolution

```text
Engineer prototype
    → shared-server pilot
    → capacity and security validation
    → IT-managed production service
    → multi-instance scaling and plant rollout
```

The public repository documents this architecture without exposing private infrastructure, controlled company terminology, or production data.
