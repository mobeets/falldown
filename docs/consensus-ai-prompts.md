# Consensus.ai Prompts — Neuronal Data × Decision Making

Copy/paste these into consensus.ai to find relevant journal articles. Each
prompt is self-contained: give it the task description, what data you have,
and the specific question you want the literature to answer.

**Task description to paste into any prompt (context):**
> Humans play a custom video game ("Falldown"): a ball falls down a screen and
> the participant steers it left/right to pass through holes in each level.
> Levels come in repeating 1-2-1 sequences: a single entry hole, then a level
> with two holes (a binary choice), then a single exit/goal hole. Choosing the
> "wrong" hole on the choice level can cost time. We recorded from
> intracranial depth electrodes in an epilepsy-monitoring-unit patient
> (Blackrock NSP, 30 kHz) while they played. The electrodes are bilateral
> mesial temporal lobe depth leads — 8 leads × 8 contacts = 64 channels — with
> contacts in the hippocampus (CA fields, anterior hippocampus, hippocampal
> body) and amygdala. Spike-sorted single units (122 units after QC) are
> aligned to the choice moment (the moment the ball passes through one of the
> two holes on the choice level), with a ±2 s window per trial. We have ~910
> trials, plus behavioral covariates per trial: which hole was chosen
> (left/right), reaction time, the greedy (1-step) vs planning (2-step) cost
> structure of each trial, and whether the two strategies agree or conflict on
> that trial.

---

## Prompt 1 — Choice selectivity & PSTH methodology

> For the task and data described above, what is the canonical published
> methodology for detecting **choice-selective neural activity** in
> single-neuron (spike) recordings from a sequential decision-making task?
> Specifically:
> - Standard PSTH construction (bin width, smoothing kernel, baseline
>   normalization / z-scoring) for low-firing-rate units (< 5 Hz).
> - Statistical tests for "choice selectivity" — comparing firing across
>   trials where the participant chose left vs right (or chose the greedy hole
>   vs the planning hole). Which papers established these approaches
>   (e.g., in monkey choice tasks, human single-unit/ECoG decision studies)?
> - Minimum trial counts / units needed to reliably detect a choice-selective
>   response at these firing rates.
> Please surface the key methods papers and any consensus best practices, with
> a focus on hippocampal and amygdala single-unit recordings (human intracranial
> and monkey), since our electrodes are entirely in the mesial temporal lobe.

## Prompt 2 — Greedy vs planning / conflict-encoding neurons

> In the task above, each choice level has a **greedy (1-step optimal) hole
> and a planning (2-step optimal) hole**. The two strategies sometimes agree
> on the same hole and sometimes conflict (each points to a different hole).
> I want to know whether and where neurons encode **strategy conflict** and
> **planning depth**. What does the literature say about:
> - Neural signatures of **conflict** in decision making (e.g., ACC/pre-SMA,
>   error/conflict monitoring) — canonical papers and the metrics used — and
>   any evidence of conflict-related signals in hippocampus or amygdala.
> - Neural correlates of **model-based vs model-free / multi-step planning**
>   in the **hippocampus** (and amygdala), and how "planning horizon"
>   has been decoded from neural data.
> - Hippocampal representation of sequential structure, successor representation,
>   and forward/replay-based planning during sequential decision tasks, since
>   our recordings are exclusively mesial temporal (hippocampus + amygdala).
> - Experimental designs that manipulate the *agreement/disagreement* between
>   a greedy and a goal-directed policy — are there existing analogues?
> Please return the most relevant empirical papers and the specific firing
> metrics (difference-of-firing, index of selectivity, etc.) they use.

## Prompt 3 — Strategy switching & latent-state neural dynamics

> We already fit a GLM-HMM to the *behavioral* data (Ashwood et al. 2022,
> "Mice alternate between discrete strategies") and found discrete
> strategies. I now want to test whether the **neural population** tracks the
> latent strategy state and whether neural activity can *predict* or *precede*
> a strategy switch. What does the literature offer for:
> - Methods to align or correlate **single-unit/population spike activity with
>   a latent behavioral state** (HMM state posteriors, hidden-state decoding).
> - Papers showing neurons track task states / strategy switches (e.g., in
>   reversal learning, attention switches, or strategy switching tasks) —
>   especially in hippocampus/amygdala.
> - Whether population activity at trial onset can decode which strategy will
>   be used on that trial (decoding methods, dimensionality reduction like
>   PCA/t-SNE/CCA on trial-aligned spike counts).
> Please give the canonical references and concrete analysis recipes, with
> attention to hippocampal/amygdalar single-unit population decoding.

## Prompt 4 — Single-trial choice decoding from population spiking

> I have trial-aligned spike counts (25 ms bins, ±2 s around choice, 122 units
> from hippocampus and amygdala, 910 trials). I want to **decode the binary
> choice** (left/right hole) from the population on single trials, and compare
> decoding accuracy across time windows around the choice. What are the
> established approaches and pitfalls in the literature?
> - Decoder choices: logistic regression / linear SVM vs LDA/QDA vs
>   time-varying / recurrent decoders; feature engineering (bin sums, rates,
>   z-scored).
> - Cross-validation schemes appropriate for sequential trial data (avoiding
>   temporal leakage).
> - How many units / trials are typically needed, and how to report chance
>   level and confidence intervals.
> - Papers that decode choice from human intracranial single/multi-unit or
>   ECoG signals and benchmark these decoders, particularly any decoding
>   choice/task variables from hippocampal or amygdala units.

---

**Usage tips**
- consensus.ai works best with one focused question at a time — run the
  prompts separately.
- Paste the "Task description" block before the specific question for context.
- If you want to scope to a subfield (human intracranial, macaque, rodent,
  computational modeling), add one line: "Restrict to [human iEEG / nonhuman
  primate / computational modeling] studies."
- Because our electrodes are all in the mesial temporal lobe (hippocampus +
  amygdala), most of these prompts already bias toward hippocampal studies;
  drop that bias if you want a broader comparison population.
