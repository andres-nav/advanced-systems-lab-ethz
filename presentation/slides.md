---
theme: seriph
title: "Interval-Masked Streaming Top-k Pearson Correlation — Final"
layout: center
class: text-center
transition: slide-left
---

<div class="text-xs tracking-[0.4em] uppercase opacity-60 mb-4">Advanced Systems Lab · Team 41 · Final</div>

# Interval-Masked Streaming Top-k Pearson Correlation

<div class="text-2xl mt-4 font-light opacity-90">
Recap on memory hierarchy optimizations<br>
<span class="text-emerald-300 font-medium">+ memory <i>layout</i> sorting optimization</span>
</div>

<div class="mt-10 flex gap-8 text-sm opacity-70 justify-center">
  <div><span class="text-emerald-300 font-bold text-lg"><Math eq="0.06 \to\; \approx 10 "/></span> <Math eq="\, \text{FLOPs}/\text{cyc}\; (\texttt{LONG})"/></div>
  <div><span class="text-emerald-300 font-bold text-lg"><Math eq="\approx 16 "/></span> <Math eq="\, \text{FLOPs}/\text{cyc}\; (\texttt{SHORT})"/></div>
  <div><span class="text-emerald-300 font-bold text-lg"><Math eq="9\;"/></span> incremental improvements</div>
</div>

<style>
.slidev-layout { background: linear-gradient(to bottom right, #0f172a, #1e293b, #1e1b4b); }
.slidev-layout h1, .slidev-layout .text-2xl, .slidev-layout div { color: white; }
</style>

<!--
Welcome to our final presentation. In meeting one we set up the problem and the
baseline; in meeting two we walked through seven architectural optimizations.
Today we close the loop: a quick recap of the journey, and then the headline new
contribution — sorting the feature rows by their validity interval to attack the
one structure we hadn't yet exploited: the overlap geometry itself.
-->

---

# Recap

<div class="max-w-3xl mx-auto text-left mt-6 text-lg leading-relaxed">

Matrix <Math eq="X \in \mathbb{R}^{F \times S}"/> — each feature <Math eq="i"/> valid only on <Math eq="[\text{start}_i, \text{end}_i)"/>.

<v-clicks>

- For every feature, find the <Math eq="K=16"/> others with the largest <Math eq="\rho_{I_i[ ... ],\, I_j[ ... ]}"/>
- Over overlap <Math eq="\;L=\max(\text{start}_i,\text{start}_j),\; R=\min(\text{end}_i,\text{end}_j)"/>
- <Math eq="\text{corr}(i,j) = \dfrac{1}{R-L-1}\textstyle\sum_{t=L}^{R-1} z_i(t)\,z_j(t)"/>
- R1: never materialize the full <Math eq="F\times F"/> matrix → streaming top-<Math eq="K"/> heap

</v-clicks>

</div>

<div class="mt-8 font-mono text-sm flex justify-center">

<pre>
Feature i:  <span class="text-blue-400">|-------- valid --------|</span>
Feature j:        <span class="text-green-400">|-------- valid --------|</span>
                  <span class="text-orange-400">|---- overlap ----|</span>
                  <span class="text-orange-400">^^^^^^^^^^^^^^^^^^^</span>
</pre>

</div>

<!--
A one-slide recap of the spec. F features, S samples, each feature valid on a
sub-interval. For each feature we want its sixteen most correlated partners, but
correlation is measured only on the shared overlap of the two intervals. And rule
one forbids storing the whole correlation matrix — we keep a size-K min-heap per
feature and update it online.
-->

---

# 3 phases, 1 bottleneck

<div class="grid grid-cols-3 gap-4 mt-6">
  <div class="p-4 rounded-lg border border-sky-500/40 bg-sky-500/5">
    <div class="text-sky-400 font-bold mb-1">Phase 1: Normalize</div>
    <div class="text-sm opacity-80"><Math eq="\mu_i,\sigma_i"/> per feature → <Math eq="z"/>-scores, once</div>
    <div class="mt-3 text-xs font-mono opacity-50">3.4% of cycles</div>
  </div>
  <div class="p-4 rounded-lg border-2 border-red-500/60 bg-red-500/10 shadow-lg shadow-red-500/20">
    <div class="text-red-400 font-bold mb-1">Phase 2: Correlate</div>
    <div class="text-sm opacity-80"><Math eq="z"/>-score dot product over every overlap <Math eq="[L,R)"/></div>
    <div class="mt-3 text-xs font-mono text-red-300 font-bold">96.6% of cycles</div>
  </div>
  <div class="p-4 rounded-lg border border-orange-500/40 bg-orange-500/5">
    <div class="text-orange-400 font-bold mb-1">Phase 3: Top-K</div>
    <div class="text-sm opacity-80">min-heap insert per pair</div>
    <div class="mt-3 text-xs font-mono opacity-50">0.1% of cycles</div>
  </div>
</div>

<div class="grid grid-cols-2 gap-8 mt-8 items-center">
<div>

```cpp
// Phase 2 inner loop
float acc = 0.0f;
for (uint32_t t = L; t < R; ++t)
    acc += Z[i*S + t] * Z[j*S + t];
corr = acc / (R - L - 1);
```

</div>
<div v-click class="text-sm">

Microbenchmark (PAPI, <Math eq="F{=}128,\, S{=}8192"/>):  
Phase 2 = **96.6%** of cycles

<div class="mt-3 text-emerald-600 font-medium">
Optimizations all focus on this dot product.
</div>

</div>
</div>

<div v-click class="absolute bottom-3 left-8 right-8 text-xs opacity-55 border-t border-slate-500/20 pt-1">
<b>Measurement:</b> growing warm-up batches prime branch predictors &amp; reach steady state, then every input buffer + scratch is <code>CLFLUSH</code>+<code>MFENCE</code>'d before each timed call — <b>numbers start from a cold cache</b>.
</div>

<!--
Profiling the algorithmic baseline with hardware counters settled the question
before we wrote any SIMD. Phase two — the pairwise dot product over each overlap
— is 96.6 percent of the cycles. Normalization and the heap are noise. So every
optimization that follows targets exactly this loop.
-->

---
layout: center
class: text-center
---

<div class="text-xs tracking-[0.3em] uppercase opacity-50 mb-2">Meetings 1 & 2 — recap</div>

# Incremental optimizations

<div class="max-w-3xl mx-auto text-left mt-4 space-y-1.5 text-base">
  <div v-click class="p-2 rounded bg-slate-500/10 border-l-4 border-slate-500 flex justify-between items-center"><span><b>V0 Baseline</b> <span class="opacity-60 text-sm ml-2">naive reference, <code>-O3</code></span></span><span class="text-xs font-mono opacity-40">0.2×</span></div>
  <div v-click class="p-2 rounded bg-slate-500/10 border-l-4 border-slate-400 flex justify-between items-center"><span><b>V1 Logical Baseline</b> <span class="opacity-60 text-sm ml-2">pre-normalize · upper-triangle · min-heap <code>-O0</code></span></span><span class="text-xs font-mono opacity-50">1×</span></div>
  <div v-click class="p-2 rounded bg-blue-500/10 border-l-4 border-blue-500 flex justify-between items-center"><span><b>V2 Auto-vectorization</b> <span class="opacity-60 text-sm ml-2"><code>-O3 -march=native</code></span></span><span class="text-xs font-mono text-blue-700">2.6×</span></div>
  <div v-click class="p-2 rounded bg-cyan-500/10 border-l-4 border-cyan-500 flex justify-between items-center"><span><b>V3 Fast-math + ILP</b> · <b>V4 Manual ILP</b> <span class="opacity-60 text-sm ml-2">reorder + unroll</span></span><span class="text-xs font-mono text-cyan-700">~3×</span></div>
  <div v-click class="p-2 rounded bg-teal-500/10 border-l-4 border-teal-500 flex justify-between items-center"><span><b>V5 Manual AVX2</b> <span class="opacity-60 text-sm ml-2"><Math eq="\small 8\; \text{FMAs} / \text{inst}"/></span></span><span class="text-xs font-mono text-teal-700">19×</span></div>
  <div v-click class="p-2 rounded bg-green-500/10 border-l-4 border-green-500 flex justify-between items-center"><span><b>V6 Reg. blocking 4×1</b> · <b>V7 Multi-acc 4×2</b> <span class="opacity-60 text-sm ml-2">hide FMA lat <Math eq="\small \to \; \approx 10\, \text{FLOPs}/\text{cyc}"/></span></span><span class="text-xs font-mono text-green-800 font-bold">47×</span></div>
  <div v-click class="p-2.5 rounded bg-emerald-500/20 border-l-4 border-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.35)] flex justify-between items-center"><span><b>V8 Sorting</b> <span class="text-sm ml-2 text-emerald-900">order feat. by interval overlap &nbsp; <b>new contribution</b></span></span><span class="text-xs font-mono text-emerald-900 font-bold">53×</span></div>
</div>

<div class="text-center text-xs opacity-50 mt-3">cumulative speedup vs logical baseline V1 · <code>LONG_GAUSSIAN</code>, <Math eq="F{=}2048,\,S{=}512"/></div>

<!--
Here is the whole ladder. Variants zero through seven are the story of meetings
one and two: an algorithmic rewrite, then climbing the micro-architecture —
auto-vectorization, fast-math, manual unrolling, explicit AVX2, register
blocking, and finally eight accumulators to saturate the two-per-cycle FMA
throughput. That got us to about ten flops per cycle, a third of the AVX2-FMA
peak. Variant eight, sorting, is what's new today.
-->

---

# Improving from V7

<div class="grid grid-cols-5 gap-6 items-center">
<div class="col-span-3">
  <img src="/perf_step8.svg" class="w-full rounded bg-white/5 p-2" />
</div>
<div class="col-span-2 text-sm space-y-3">

<div class="p-3 rounded bg-green-500/10 border border-green-500/30">
<b class="text-green-800">V7 = 8 accumulators (4×2)</b><br>
Saturates the 2 FMA ports → <br><Math eq="\small \approx 10 \text{FLOPs}/\text{cyc} \approx \tfrac{1}{3}\pi_\text{peak}"/>
</div>

<v-clicks>

- Inner loop close to optimal (microarch.)
- But touches **every feature pair** and re-walk overlaps blindly
- Dip at <Math eq="S{=}2^{14}"/>: &nbsp; <Math eq="Z = 8\text{MB} > 6\text{MB}"/> L3 → DRAM-bound

</v-clicks>

<div v-click class="mt-2 p-3 rounded bg-emerald-500/10 border border-emerald-500/30">
Untouched lever: the <b>overlap structure</b>.<br>
What if we reorder the data to exploit it?
</div>

</div>
</div>

<!--
This is where meeting two ended: V7, the eight-accumulator kernel, riding at
about ten flops per cycle. The loop itself is essentially maxed for this data
layout. But two things are still on the table. First, we blindly visit every
pair and recompute its overlap bounds. Second, the rows are in arbitrary order,
so consecutive pairs have unrelated overlaps and unrelated cache footprints.
Both are properties of the data layout, not the loop — so let's change the
layout.
-->

---
layout: center
class: text-center
---

<div class="text-xs tracking-[0.3em] uppercase text-emerald-400 mb-3">New improvement</div>

# V8: Sort features by interval

<div class="text-xl font-light opacity-80 max-w-2xl mx-auto mt-4">
Permute rows <Math eq="\to"/> <code>start[i]</code> monotone<br>
2 big benefits
</div>

<!--
The new idea is almost embarrassingly simple: sort the feature rows by their
interval start. One std::sort up front. Everything downstream operates on the
permuted, sorted order. That single reordering unlocks two distinct wins.
-->

---

# Why sorting pays off

<div class="grid grid-cols-2 gap-8 mt-4">

<div>
<div class="font-bold mb-2">1. Monotone starts → early exit</div>

Sorted <code>start[]</code> non-decreasing.<br>
For <Math eq="i"/>-block with <Math eq="\small E = \max \text{end}[i]"/>:<br>&nbsp;&nbsp;<Math eq="\small \text{start}[j] \;\ge\; E \Rightarrow"/> no more overlap<br>&nbsp;&nbsp;<Math eq="\to"/> <code>break</code> loop

```cpp
uint32_t E = max(end[i..i+3]);
for (j = i+4; ...; j += 2) {
    if (starts[j] >= E) break; // exit
    /* 8-acc 4x2 dot products */
}
```

<div class="text-sm opacity-70 mt-2">In <code>GAUSSIAN_SHORT</code>, most disjoint → skipped</div>
</div>

<div>
<div class="font-bold mb-2">2: Overlap cluster → cache locality</div>

Adjacent intervals similar <Math eq="\small [L,R)"/>:

<v-clicks>

- cluster by overlap → strip of <Math eq="Z"/> hot in L1/L2
- <code>L … R</code> tail loop shrink (less redundancy)
- contiguous sorted rows  
  &nbsp;<Math eq="\to"/> prefetch-friendly strides

</v-clicks>

<div v-click class="mt-2 font-mono text-xs text-slate-400">
<pre>
Unsorted:
    Iter 1: <span class="text-rose-400">[===]</span>
    Iter 2:            <span class="text-rose-400">[===]</span>
    Iter 3:      <span class="text-rose-400">[===]</span>
    Iter 4:                 <span class="text-rose-400">[===]</span>
Sorted:
    Iter 1: <span class="text-emerald-400">[===]</span>
    Iter 2:  <span class="text-emerald-400">[===]</span>
    Iter 3:   <span class="text-emerald-400">[===]</span>
    Iter 4:    <span class="text-emerald-400">[===]</span>
</pre>
</div>

</div>

</div>

<!--
Two consequences. First, because sorted starts are monotone, the moment a
candidate j starts after the latest end of the current i-block, every later j is
disjoint too — so we break the entire inner loop instead of testing each pair.
In the short-interval regime that prunes the vast majority of pairs. Second,
neighbouring sorted rows have similar overlap windows, so the slice of Z we
actually touch stays resident in cache across consecutive pairs, and the ragged
peel-and-tail handling for the AVX edges shrinks. The first effect cuts work; the
second cuts memory traffic.
-->

---

# V8 permutation implementation

<div class="grid grid-cols-2 gap-6 mt-2 text-sm">
<div>

```cpp
// step 1: sort by start
for (i) perm[i] = i;
std::sort(perm, perm+F, [&](a,b){start[a] < start[b]});
for (si) {              // cache sorted bounds
    sorted_starts[si] = starts[perm[si]];
    sorted_ends[si]   = ends[perm[si]];
}
```

```cpp
// step 2: normalize, read original row
//          perm[si], write sorted row si
Z[si*S + t] = (X[perm[si]*S+t] - mean) * inv_std;
```

</div>
<div>

```cpp
// step 3: sorted-space dot-prod: orig. idx → heap
heap_insert(&heaps[i*K], v, perm[j]);
```

```cpp
// step 4: scatter through perm
heap_to_topK(&heaps[si*K], &topK[perm[si]*K]);
```

<div class="mt-4 p-3 rounded bg-slate-500/10 border border-slate-500/30">
Output identical to V7; permutation only internal. Early-exit reduces FLOPs – depends on domain.
</div>

</div>
</div>

<!--
The only cost of sorting is bookkeeping. Phase one reads the original row but
writes the sorted slot, so phase two runs entirely in sorted space with
contiguous rows. The heap stores original feature indices — we map through perm
at insertion — and phase three scatters each heap back to its original row. The
output is bit-for-bit the same top-K as V7; sorting is a pure internal layout
change, so the comparison stays honest.
-->

---

# `LONG_GAUSSIAN` (default)

<div class="text-sm opacity-60 mb-2">Performance vs <Math eq="F"/> (left, <Math eq="S{=}512"/>) and vs <Math eq="S"/> (right, <Math eq="F{=}512"/>)</div>

<div class="grid grid-cols-4 gap-2 items-center">
  <div class="col-span-2">
  <img src="/perf_F_step9.svg" class="w-full rounded bg-white/5 p-1" />
    <div v-click="4" class="absolute left-[25%] bottom-[32%] px-3 py-1 rounded bg-green-500/10 border border-green-500/30 text-sm">
      <!-- V7 <b class="text-slate-700"><Math eq="\approx 12\times"/></b> vs V1 -->
      <b class="text-green-800">V7 <Math eq="\approx 47\times"/></b> <span class="opacity-60">vs V1</span>
    </div>
  </div>
  <div class="col-span-2"><img src="/perf_step9.svg" class="w-full rounded bg-white/5 p-1" />
    <div v-click="5" class="absolute right-[15%] bottom-[32%] px-3 py-1 rounded bg-emerald-500/15 border border-emerald-500/40 text-sm">
      <b class="text-emerald-800">V8 <Math eq="\approx 53\times"/></b> vs V1
    </div>
  </div>
</div>

<!-- <div class="flex justify-center gap-6 mt-2 mb-1"> -->
<!--   <div class="px-3 py-1 rounded bg-green-500/10 border border-green-500/30 text-sm"> -->
<!--   <b class="text-green-800">V7 ≈ 47×</b> <span class="opacity-60">vs V1</span> -->
<!--   </div> -->
<!--   <div class="px-3 py-1 rounded bg-emerald-500/15 border border-emerald-500/40 text-sm"> -->
<!--   <b class="text-emerald-800">V8 ≈ 53×</b> <span class="opacity-60">vs V1</span> -->
<!--   </div> -->
<!-- </div> -->

<v-clicks>

- `LONG`: 80% intervals <Math eq="\to"/> most overlap
- Early-exit rare
- Cache locality <Math eq="\to"/> still <Math eq="\small \approx 5–13\%"/> faster at large <Math eq="F"/>

</v-clicks>

<!--
First the default LONG-Gaussian regime, where intervals cover eighty percent of
the samples. Here almost every pair overlaps, so the early-exit hardly ever
triggers — and V8 sits right on top of V7. We still see a modest five-to-thirteen
percent cycle reduction at large F purely from the better cache behaviour, but
this regime is not where sorting shines. For that, we need short intervals.
-->

---

# `SHORT_GAUSSIAN` (sorting wins)

<div class="text-sm opacity-60 mb-2">Performance: <Math eq="F"/> (left, <Math eq="S{=}512"/>) vs <Math eq="S"/> (right, <Math eq="F{=}512"/>)</div>

<div class="grid grid-cols-4 gap-4 items-center">
  <div class="col-span-2">
    <img src="/short_gaussian_perf_F_step9.svg" class="w-full rounded bg-white/5 p-1" />
    <div v-click="[4,5]" class="absolute left-[20%] bottom-[30%] px-3 py-1 rounded bg-slate-500/10 border border-slate-500/30 text-sm">
      <b class="text-slate-700">V7 <Math eq="\approx 12\times"/></b> vs V1
    </div>
    <div v-click="5" class="absolute left-[20%] bottom-[30%] px-3 py-1 rounded bg-emerald-500/15 border border-emerald-500/40 text-sm">
      <b class="text-emerald-800">V8 <Math eq="\approx 27\times"/></b> vs V1, <b class="text-emerald-800"><Math eq="\;\approx 2.3\times"/></b> vs V7
    </div>
  </div>
  <div class="col-span-2"><img src="/short_gaussian_perf_step9.svg" class="w-full rounded bg-white/5 p-1" /></div>
</div>


<v-clicks>

- Reaches <Math eq="\small \approx 16\, \text{FLOPs}/\text{cyc}"/>
- `SHORT`: 30% intervals → mostly disjoint, early-exit prunes 
- V7 computes every pair; V8 does not — pure work removed

</v-clicks>

<!--
Now short intervals — thirty percent coverage. Here most feature pairs simply
don't overlap, and this is exactly what the monotone early-exit was built for. V8
pulls away from the whole stack and keeps climbing with F, reaching about sixteen
flops per cycle while V7 and below stay flat near one. The gap is the work we no
longer do: V7 visits every pair and discovers the empty overlap; V8 breaks out of
the loop and never touches them.
-->

---
transition: fade-out
---

# Rooflines

Operational intensity vs performance, sweep over <Math eq="F"/> &nbsp;<span class="text-sm opacity-60">(<Math eq="S{=}512"/>)</span>

<div class="flex justify-center relative h-[380px] bg-white/5 rounded p-2">
  <img v-click.hide="1" src="/roofline_F_step2.svg" class="absolute inset-0 h-full w-full object-contain transition-opacity duration-300" />
  <img v-click="[1,2]" src="/roofline_F_step5.svg" class="absolute inset-0 h-full w-full object-contain transition-opacity duration-300" />
  <img v-click="[2,3]" src="/roofline_F_step6.svg" class="absolute inset-0 h-full w-full object-contain transition-opacity duration-300" />
  <img v-click="[3,4]" src="/roofline_F_step8.svg" class="absolute inset-0 h-full w-full object-contain transition-opacity duration-300" />
  <img v-click="4"     src="/roofline_F_step9.svg" class="absolute inset-0 h-full w-full object-contain transition-opacity duration-300" />

  <div v-click.hide="1" class="absolute right-[0%] bottom-[18%] bg-orange-600/85 text-orange-50 p-2 rounded text-xs border border-orange-500 max-w-[210px]">
    <b>V1:</b> logical baseline, far below rooflines.
  </div>
  <div v-click="[1,3]" class="absolute right-[0%] bottom-[18%] bg-orange-900/80 text-orange-100 p-2 rounded text-xs border border-orange-900 max-w-[210px]">
    <b>V5:</b> AVX2 latency bound at <br><Math eq="\small \approx 3.5\, \text{FLOPs}/\text{cyc}"/>
  </div>
  <div v-click="[3,4]" class="absolute right-[0%] bottom-[18%] bg-slate-900/80 text-slate-100 p-2 rounded text-xs border border-slate-500 max-w-[210px]">
    <b>V7:</b> 8 accs hide FMA latency <Math eq="\to\; \approx \frac{1}{3}"/> AVX2 FMA peak
  </div>
  <div v-click="4" class="absolute right-[0%] bottom-[18%] bg-green-800/85 text-green-50 p-2 rounded text-xs border border-green-400 max-w-[230px]">
    <b>V8:</b> sorting raises block-<Math eq="I"/>-overlap <Math eq="\to"/> cache locality
  </div>
</div>

<!--
The roofline ties it together. We start as a scalar point pinned to the floor,
far below every ceiling. AVX2 lifts us off the ground but leaves us latency-bound
at three and a half. Eight accumulators push us into the compute region at about
a third of the FMA peak. And sorting, in V8, nudges operational intensity to the
right: keeping the working strip cache-resident means fewer bytes per flop, so
the same compute sits at higher intensity, further from the bandwidth diagonals.
-->

---
layout: center
class: text-center
---

# Summary

<div class="grid grid-cols-3 gap-6 max-w-4xl mx-auto mt-6 text-left">

<div class="p-4 rounded-lg bg-slate-500/10 border border-slate-500/30">
<div class="text-3xl font-bold text-slate-900">96.6%</div>
<div class="text-sm opacity-70 mt-1">of cycles one dot-product, seen in micro-benchmarks </div>
</div>

<div class="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
<div class="text-3l font-bold text-green-800"><Math eq="\approx 10 \to 16\, \small \text{FLOPs}/\text{cyc}"/></div>
<div class="text-sm opacity-70 mt-1">V7 micro-arch peak → V8 sort<br> in <code>SHORT</code></div>
</div>

<div class="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
<div class="text-3xl font-bold text-emerald-800">53× / 27×</div>
<div class="text-sm opacity-70 mt-1">V8 <code>LONG</code> / <code>SHORT</code><br> at <Math eq="\small F{=}2048"/> vs logical baseline</div>
</div>

</div>

<div class="max-w-3xl mx-auto mt-8 text-left text-base space-y-2">
<v-clicks>

- **Meetings 1–2:** micro-architectural improvements (SIMD → blocking → 8 accumulators)
- **New:** data layout improvement – sort by interval
- Regime-dependent: invisible in `LONG`, significant in `SHORT`; adaptivity adheres to R2.

</v-clicks>
</div>

<!--
To summarize: profiling sent us to a single loop; the micro-architectural climb
of meetings one and two took it to about ten flops per cycle; and today's
contribution showed that once the loop is maxed, the remaining win lives in the
data layout. Sorting rows by interval is one std::sort that buys a monotone
early-exit and a cache-resident overlap strip. It's invisible when intervals are
long and everything overlaps, but decisive when they're short — which is exactly
the adaptivity the spec invited.
-->

---
layout: center
class: text-center
---

# Questions

<div class="text-lg opacity-70 mt-2">Team 41</div>

<div class="mt-6 text-sm opacity-50 font-mono">
i7-7700HQ (Kaby Lake) · core-pinned, turbo off · PAPI harness · warm branch pred., cold cache
</div>

<style>
.slidev-layout { background: linear-gradient(to bottom right, #0f172a, #1e293b, #1e1b4b); }
.slidev-layout h1, .slidev-layout .text-lg, .slidev-layout .text-sm { color: white; }
</style>

<!--
Thank you. We're happy to take questions on the SIMD kernel, the accumulator
tuning, the sorting contribution, or the benchmarking methodology.
-->
