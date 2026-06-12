# Future Directions: Caging Loops × TRELLIS.2

A design note on combining the caging-loop pipeline (this repo) with Microsoft
**TRELLIS.2** (single-image / text → 3D, "Omni-Voxel" representation) into a new,
trained perception-to-grasp system. This captures the design discussion: what the
input is, how a loop comes out, what gets trained, and the goal.

> One-line goal: **turn "an image and/or a description of an unknown object" into a
> verified, physically-sound caging grasp, in seconds — robust to occlusion.**

---

## 1. Background — TRELLIS.2 and the O-Voxel structure

From the TRELLIS.2 paper ([arXiv 2512.14692](https://arxiv.org/html/2512.14692v1)):

- **O-Voxel (Omni-Voxel)** = a *field-free* **sparse** voxel structure. Each **active**
  voxel (only those whose edges intersect the surface) stores `(f_shape, f_mat, p)`:
  - `f_shape` = **Flexible Dual Grid** (Dual Contouring): a sub-voxel **dual vertex**
    `v∈[0,1]³`, **edge-intersection flags** `δ∈{0,1}³`, **splitting weights** `γ`.
    A boundary-aware QEF preserves sharp edges and **open** surfaces.
  - `f_mat` = **PBR**: base color, **metallic**, **roughness**, **opacity**.
- **Resolution 1024³–1536³**; SC-VAE compresses 1024³ → **~9.6K latent tokens** (16×).
- **Image→3D / text→3D**, 3-stage flow-matching DiT (sparse structure → geometry →
  material). ~3 s (512³) / ~17 s (1024³) / ~60 s (1536³) on an H100. DINOv3 conditioning.
- Mesh↔O-Voxel is optimization-free (seconds CPU / tens of ms).

**Key implication for caging:** O-Voxel is *field-free / surface-only* — it has **no
inside/outside**. Caging is intrinsically a **solid** operation (grasp space = exterior
of a solid; Thm 3.2 bounds loops between surface `S` and its hull `H`). So:

- **Closed shapes:** O-Voxel → mesh (fast decoder) → derive solid (flood-fill / winding)
  → run caging. Clean.
- **Open shapes:** caging is genuinely undefined (you can't cage an open shell) —
  consistent with the single-view finding. O-Voxel extends *past* where caging applies.

→ **O-Voxel is a high-res, topology-correct geometry *front-end*; we derive the solid
and run caging on top.** Field-free representation + field-based caging are complementary.

---

## 2. The combined system (input → loop → goal)

### Input
**A single RGB image** of an object (deployment case), **and/or a text prompt**
(intent + hidden-physics priors). Variants: RGB-D, or an existing mesh/point cloud
(skip generation).

### Output
**A caging loop** (closed 3D curve around the object) + the gripper pose to execute it.

### Flow

```
 image (geometry)  +  text (intent + hidden physics)
        │
        ▼
[1] TRELLIS.2  (frozen, pretrained)
   ├─► complete 3D asset   (O-Voxel: topology-correct, incl. the unseen back)
   ├─► structured latent   (SC-VAE tokens)
   └─► PBR materials        (roughness, metallic, opacity)  ──► physical fields (§7)
        │
        ▼
[2] Caging head  (THE NEW TRAINED PART)
        latent  ──►  predicted Morse-saddle field / saddles
        │
        ▼
[3] Trace  (classical, Thm 3.1)
        each saddle ──► two opposite shortest arcs ──► candidate loops
        │
        ▼
[4] Filters + physics ranker  (classical, exact)
        encircle/link + locally-shortest + mechanical(true-CoG + horizontal)
        + slip-margin + avoid no-touch + functional intent
        │
        ▼
   best caging loop  ──►  [5] gripper pose (origin + Dir1/Dir2)
```

- **[1] frozen TRELLIS** — image/text → complete shape + latent + materials.
- **[2] the only new trained model** — predicts *where the saddles are*, replacing the
  slow `Dp` fast-marching solve.
- **[3]–[4] existing classical code** — trace + filters guarantee validity. The network
  *proposes where to look*; it is never trusted to emit a correct curve.

### What's trained vs reused

| component | status |
|---|---|
| TRELLIS.2 (image/text→3D + latent) | **frozen, pretrained** |
| Caging head (latent → saddles/field) | **trained (distilled)** — the only new model |
| Trace (Thm 3.1), filters, ranker, gripper pose | **classical, not trained** |

### Why the combination beats either alone

| | image → grasp? | unseen back? | fast? | grasp valid? |
|---|---|---|---|---|
| Caging pipeline alone | ✗ (needs a complete solid) | ✗ | ✗ (slow solve) | ✓ |
| TRELLIS alone | — (makes 3D, no grasp) | ✓ | ✓ | — |
| **Combined trained model** | **✓** | **✓** | **✓** | **✓** |

The combination delivers three things at once: **perception-to-grasp from one image/prompt**,
**speed** (head amortizes the solve), and **guaranteed-valid output** (filters verify).

---

## 3. The five integration directions

1. **Image/text→3D as completion → solves single-view caging.** A learned, topology-aware
   "hallucinate the shape" (better than Poisson). Cage the generated asset. Safety caveat:
   it's a *generated guess*; keep the conservative occlusion-solid as an overlay / verify
   against the observed front.
2. **High-res sparse geometry breaks the resolution wall.** Every handle-threading failure
   was the grid not resolving the hole at 40–60³. Dual-Contour surface at 1024³ resolves
   thin handles → the threading saddle *exists*. The **GPU saddle + cross-base tracer** we
   built is what makes the field tractable at that resolution (dense msfm is impossible;
   sparse-on-active-voxels + GPU is the path).
3. **Sub-voxel grasp space from dual vertices.** O-Voxel stores a sub-voxel surface
   position per cell → place the grasp-space boundary and traced loops at sub-voxel
   accuracy (our field is voxel-binary).
4. **PBR → physics-weighted field, for free.** roughness→friction, metallic/material→density
   →CoG, opacity→graspable/translucent. The material map is aligned to geometry. (See §7.)
5. **Caging in the *learned* latent.** The paper computes loops in a *distance-field*
   embedding; SC-VAE is a *learned* shape embedding. Predict the caging structure with a
   head on the latent. (See §4.)

Near-term, highest value: **#2** (finally get handle loops, enabled by our GPU work) and
**#1** (single-view via completion). **#4** is a free material upgrade; **#3** a precision
refinement; **#5** the research bet.

---

## 4. Deep dive — caging in the learned latent (#5)

### Philosophy
The paper's thesis is "don't tie loops to surface triangles — work in an **embedding
space**" (their embedding = the distance field `Dp`; loops = its **Morse saddles**,
Thm 3.1). TRELLIS gives a *learned, sparse, geometry-aligned* embedding. Caging loops are
a **topological** property; the latent is **topology-aware** → the map
`shape-topology → caging-loop-topology` exists (Morse theory) and is what we learn.

### What to learn (most→least tractable)

| | learn | extraction | tractability |
|---|---|---|---|
| **A. Field** | predict `Dp` from latent + base point | classical saddles + trace | **easiest, most faithful** |
| **B. Saddles** | predict a saddle heatmap over active voxels | classical trace | easy |
| **C. Loops** | emit closed curves directly | none | **hardest, no validity guarantee** |
| **D. Ranker** | score/select among proposed loops | — | easy, complementary |

**Avoid jumping to C.** Loops are variable-count, variable-length closed curves that must
be locally-shortest, encircle material, and not self-intersect — emitting valid ones
directly has no built-in guarantee. **A/B keep the classical extraction**, so the output
follows Thm 3.1 *by construction*; you only learn the expensive part (the field).

### The sweet spot — propose–verify hybrid (A/B + exact filters)
1. **Learn the field/saddles** from the latent → amortizes the expensive per-base-point
   fast-marching solve into one forward pass.
2. **Extract loops classically** (Thm 3.1) → correct by construction.
3. **Verify with exact filters** (encircle/link, locally-shortest, mechanical/physics
   ranking) → validity guaranteed; the net only *proposes where to look*.

### Architecture sketch
- **Input:** SC-VAE tokens (sparse `[N,C]`, geometry-aligned) + base-point embedding +
  task conditioning (gripper reach `2h`, CoG/height target, no-touch mask).
- **Backbone:** sparse transformer / sparse-conv U-Net over active voxels (mirroring
  TRELLIS's sparse DiT), or a light head on a frozen TRELLIS encoder.
- **Head:** (A) per-voxel scalar = `Dp` (regress) or (B) per-voxel saddle probability
  (focal-BCE) → NMS.
- **Base-point-free variant (the dream):** predict the union over all base points — the
  caging-loop *space* `L̃` directly as a saddle density.

### What it buys
- **Speed** (one forward pass replaces N sweeps → ms).
- **Robustness** (inherits the generative prior → partial/noisy input).
- **Base-point-free** (predict the loop space).
- **Differentiable** (gradient-based grasp optimization; grasp-aware generation in the
  same latent).
- **Task-conditioned** (fold physics in as conditioning).

### Hard problems
1. Validity isn't free → only A/B + classical extraction guarantees it → propose–verify.
2. Domain gap: TRELLIS latents are trained for appearance, not caging → may need encoder
   fine-tune or latent-alongside-geometry.
3. Gripper-scale conditioning on `2h`; train across scales.
4. Label quality: the student can't exceed the teacher (it can denoise it).
5. Variable cardinality (C/D) → set prediction (Hungarian) or heatmap+NMS (B avoids it).

---

## 5. Training design — distilling the pipeline into a latent head

### Reframe: the pipeline IS the zero-shot teacher
The exact pipeline (msfm + GPU saddle + tracer + filters) is a **deterministic geometric
algorithm — no training, runs on any object**. So:

- **Teacher** = the exact pipeline (zero-shot, correct, slow per-asset).
- **Student** = the caging head on TRELLIS's latent (fast, robust, differentiable).
- **Training = distillation** — the student reproduces the teacher's outputs.
- **TRELLIS's two roles:** (1) frozen **feature backbone** the student reads;
  (2) a **data generator** that expands the asset distribution.

→ You are **distilling a known algorithm into a fast latent predictor**, not learning
grasping from scratch. Training buys speed/robustness/differentiability/conditioning,
**not correctness** (the teacher already has that).

### The three regimes
- **Distillation (main):** labels `(latent, base point) → saddle field`; student loss =
  field MSE (A) / focal-BCE saddle heatmap (B) + optional loop-consistency (Chamfer on
  traced loops). Classical trace + exact filters at inference (propose–verify).
- **Few-shot:** freeze the TRELLIS encoder, train only a **light head** → few labeled
  assets adapt to a new domain. Works because the latent already encodes the topology.
- **Zero-shot:** *already have it* — run the teacher (no head), optionally on a
  TRELLIS-completed single-view object. Deployable today, no training. *Learned* zero-shot
  transfer to unseen categories rides TRELLIS's prior — verify, don't assume.

### "Use TRELLIS output to train" — the data engine
1. **Asset pool** = Objaverse (encode via TRELLIS) **+ TRELLIS-generated** (image/text→3D,
   matching the deployment distribution + unlimited rare categories).
2. For each asset → encode to latent + **run the GPU teacher** → dense labels. Our GPU
   speedups are what make labeling at *scale* feasible (else this is the bottleneck).
3. Student trains on `(latent, p) → teacher labels`; it can even **denoise** the teacher.

### Staged plan (de-risked)
- **Stage 0 — Probe before committing.** Encode a few hundred assets; run the teacher; fit
  a **linear probe** `latent → saddle probability`. Measure saddle IoU / `Dp` correlation
  (reuse `saddle_set_divergence` + loop-area metrics). *Linear probe predicts well → few-shot
  is wide open; if not → fine-tune the encoder or add capacity.* Cheap go/no-go test.
- **Stage 1 — Distill the field/saddles (A→B).** Frozen encoder + sparse head; match teacher
  zero-shot on held-out.
- **Stage 2 — Few-shot domain adaptation.** New category → fine-tune only the head.
- **Stage 3 — Task/physics conditioning.** Condition on `2h`, CoG, friction, no-touch →
  predict task-appropriate loops; supervise with teacher runs under those params.
- **Inference:** image/text → TRELLIS asset+latent → student saddles → classical trace →
  exact filters verify → loop.

### Label-cost reducers (semi-supervised)
- **Consistency / self-distillation:** same asset at different resolutions/views → loops
  must agree → augmentation loss, fewer labels.
- **TRELLIS variant generation:** variants of an asset → cheap related-structure augmentation.

---

## 6. Text prompt — dissolving the occlusion problem + functional intent

Don't over-invest in reconstructing the occluded back: the back was only a problem because
we insisted on using *only the observed partial scan*. With a **generative prior** (image
*or* text), you always get a complete object.

**What text changes:**
- **No occlusion at all** — text→3D generates a complete watertight asset; no unseen back,
  no occlusion-solid hack.
- **Use-case shift:** image → grasp *this specific* object (geometry matches reality);
  text → grasp an object matching a *description*, generate *training assets*, or
  grasp-by-description.
- **Semantic / functional channel** (the paper explicitly lacks this): *"grasp the mug by
  the handle"*, *"hold the knife by the handle, not the blade"* → condition the
  head/ranker on **intent** → pick the functionally correct loop.
- **Hidden-physics priors** (see §7): *"heavy steel hammer"*, *"empty plastic bottle"* →
  the mass distribution you can't see from geometry.
- **Best = multi-modal:** image (true geometry) + text (intent + hidden physics).
- **Caveat:** a generated shape is a guess. For the real object, condition on the image and
  **verify contacts against the observed front**. For planning/data/by-description, text alone is fine.

---

## 7. Physical characteristics — CoG, friction, no-touch

A complete generated solid makes physics tractable. Three sources feed the physical fields:

| field | source |
|---|---|
| **friction map** μ(x) | TRELLIS **roughness** channel |
| **density / mass** ρ(x) → **CoG** | TRELLIS **metallic/material** + **text** ("heavy"/"hollow"); else uniform |
| **no-touch / fragile / hot** | **text** ("don't touch the blade"), or thermal/segmentation input |
| **opacity / graspable** | TRELLIS **opacity** channel |

**True CoG:** with the generated solid + density ρ, `CoG = ∫ρ·x / ∫ρ` over solid voxels —
the *mass-weighted* center, not the geometric centroid. Text supplies the invisible part
("the head is heavy" → ρ concentrated there → CoG shifts).

**Two places physics enters (keep them separate):**
- **(a) Routing — task-weighted field.** Friction/no-touch are *local* → bake into the
  distance-field cost: ∞-cost at no-touch zones (loops route around), low cost near
  high-friction contacts (loops prefer them). *Changes which loops exist.*
- **(b) Balance — selection/ranking.** CoG/mass are *global* (a property of the whole loop)
  → belong in the **ranker**, not the field. This is the `--select mechanical` we built
  (loop center near CoG + horizontal), now with the **true mass-weighted CoG** + a
  **slip margin** (does μ·contact-force beat the gravity torque = mass × lever-arm?).

**In the trained model:** condition the head on friction/no-touch (distill a teacher that
*used* the task-weighted field), and/or keep physics in the **classical ranker** (simpler,
exact). Net result — the chosen loop is: geometrically valid (link + locally-shortest),
physically sound (near true CoG, horizontal, positive slip margin, avoids slippery/no-touch
regions), and functionally appropriate (text intent).

---

## 8. Honest caveats & risks

- **Generated shape ≠ exact object.** Image-condition + verify against observation for the
  real object; text-only is for priors/data/description.
- **O-Voxel gives geometry, not a solid.** Derive inside/outside; caging only applies where
  the shape is closed.
- **The student can't exceed the teacher** — it inherits (at best denoises) our heuristics.
  Keep the **exact filters as a hard validity gate** so a student error never emits an
  invalid grasp.
- **Domain gap** (TRELLIS latent trained for appearance, not caging) — Stage 0's probe is
  the cheap test.
- **fp32 / approximations** in the GPU path shift results slightly (measured ~1e-5 on loop
  areas); the divergence report keeps "a little" honest.

---

## 9. Roadmap

1. **Now (no training):** TRELLIS image/text→3D completion → exact pipeline (GPU-accelerated)
   → loops. Deployable zero-shot system. Use it as the **data engine**.
2. **Stage 0:** linear-probe the latent for the caging signal (go/no-go).
3. **Stage 1:** distill `Dp`/saddles into a head on the frozen latent; propose–verify.
4. **Stage 2:** few-shot domain adaptation (head-only).
5. **Stage 3:** physics + functional-intent conditioning (CoG, friction, no-touch, text).
6. **Research:** base-point-free `L̃` prediction; differentiable caging-in-latent →
   grasp optimization and grasp-aware generation.

---

## Sources
- [TRELLIS.2 paper (arXiv 2512.14692v1)](https://arxiv.org/html/2512.14692v1)
- [TRELLIS.2 GitHub](https://github.com/microsoft/TRELLIS.2) ·
  [project page](https://trellis2.netlify.app/) ·
  [TRELLIS.2-4B (HuggingFace)](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [Trellis 2: Omni-Voxels (blog)](https://blog.datameister.ai/trellis-2-controlled-3d-generation)
- [TRELLIS 1 paper (arXiv 2412.01506)](https://arxiv.org/html/2412.01506) ·
  [project page](https://microsoft.github.io/TRELLIS/)
- Caging-loop method: Liu et al., *Caging Loops in Shape Embedding Space: Theory and
  Computation* (this repo's reference paper).
