I’m reconstructing the average-linkage tree around the `4`-cluster level so I can name the first downstream split decisions precisely and separate them from the deeper cascade they trigger.

Ran terminal command:  awk -F, 'NR>1 && $4=="False" {printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1,$3,$18,$19,$12,$16,$22,$23}' 'benchmarks/results/run_20260324_171243Z/audit/case_2_kl_divergence_stats.csv' | sort -t, -k2,2nr -k1,1

Ran terminal command:  awk -F, 'NR>1 && $4=="False" {printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1,$3,$18,$19,$12,$16,$22,$23}' 'benchmarks/results/run_20260324_171243Z/audit/case_2_kl_(single)_stats.csv' | sort -t, -k2,2nr -k1,1

**First Extra Splits**
If we take the natural 4-cluster frontier in the average-linkage tree to be the four size-10 subtrees under the two size-20 branches, then the first extra splits are these.

The 4-cluster frontier itself is visible at:
- case_2_kl_divergence_stats.csv
- case_2_kl_divergence_stats.csv
- case_2_kl_divergence_stats.csv
- case_2_kl_divergence_stats.csv

So the average-linkage tree already fails at the exact point where it should stop at 4. Three of the four size-10 blocks keep splitting.

The first five extra splits are:

1. N71, size 10, corrected sibling p-value about $3.5 \times 10^{-291}$, so it splits into a size-3 child and a size-7 child.
   - split decision: case_2_kl_divergence_stats.csv
   - children: case_2_kl_divergence_stats.csv and case_2_kl_divergence_stats.csv

2. N72, size 10, corrected sibling p-value about $8.25 \times 10^{-112}$, so it splits into a size-2 child and a size-8 child.
   - split decision: case_2_kl_divergence_stats.csv
   - children: case_2_kl_divergence_stats.csv and case_2_kl_divergence_stats.csv

3. N74, size 10, corrected sibling p-value about $2.88 \times 10^{-298}$, so it splits into a size-3 child and a size-7 child.
   - split decision: case_2_kl_divergence_stats.csv
   - children: case_2_kl_divergence_stats.csv and case_2_kl_divergence_stats.csv

At that point you have already gone from the intended 4 clusters to 7.

Then the cascade starts immediately:

4. N68, size 7, corrected sibling p-value about $2.42 \times 10^{-285}$, so it splits into size 3 and size 4.
   - split decision: case_2_kl_divergence_stats.csv
   - children: case_2_kl_divergence_stats.csv and case_2_kl_divergence_stats.csv

5. N69, size 8, corrected sibling p-value about $7.84 \times 10^{-298}$, so it splits into size 5 and size 3.
   - split decision: case_2_kl_divergence_stats.csv
   - children: case_2_kl_divergence_stats.csv and case_2_kl_divergence_stats.csv

Those five extra splits are enough to explain the direction of failure. After them, more descendants such as case_2_kl_divergence_stats.csv, case_2_kl_divergence_stats.csv, case_2_kl_divergence_stats.csv, and case_2_kl_divergence_stats.csv continue the same pattern until the run ends at 26 clusters.

**Why This Is The Problem**
The underlying problem is not “the root split is wrong.” The root is fine in all three methods:
- average: case_2_kl_divergence_stats.csv
- single: root row_stats.csv#L80)
- diffusion: root row_stats.csv#L80)

The actual problem is this:

Average linkage keeps producing locally “clean” subtrees below the correct 4-cluster scale, and the same data are then used to test those subtrees. That creates a post-selection feedback loop. In a case with 20,000 binarized features, even weak local distortions become highly detectable once the tree has already arranged the samples into favorable contrasts.

A telling signal is that many of these extra splits have tiny branch lengths but enormous significance. For example:
- case_2_kl_divergence_stats.csv has branch length about $3.5 \times 10^{-4}$ yet corrected sibling p-value about $2.42 \times 10^{-285}$
- case_2_kl_divergence_stats.csv has branch length about $1.9 \times 10^{-4}$ yet corrected sibling p-value about $7.84 \times 10^{-298}$
- case_2_kl_divergence_stats.csv has branch length about $4.6 \times 10^{-4}$ yet corrected sibling p-value about $3.26 \times 10^{-298}$

That is the signature of a tree that is over-resolving noise-like local structure.

**What Single Linkage Gets Right**
Single linkage does not stop at the root either. It also accepts the root and the two size-20 splits:
- N76_stats.csv#L78)
- N77_stats.csv#L79)

But at the 10-leaf frontier it mostly stops:
- N70, skipped_stats.csv#L72)
- N73, merge_stats.csv#L75)
- N74, merge_stats.csv#L76)
- N75, merge_stats.csv#L77)

So single linkage gives the right qualitative pattern:
strong significance at coarse levels, then loss of significance at fine levels.

**What Diffusion Adds**
Diffusion is useful because it shows the same failure mode can happen with a different tree shape.

Its benchmark row is full_benchmark_comparison.csv. Like average linkage, it ends at 26 clusters.

But diffusion does not fail in exactly the same place:
- at size 10, only N73_stats.csv#L75) still splits strongly
- N71_stats.csv#L73), N74_stats.csv#L76), and N75_stats.csv#L77) do not

But then the significance persists one level lower on descendants such as:
- N72, size 9, split_stats.csv#L74)
- N70, size 9, split_stats.csv#L72)
- N68, size 6, split_stats.csv#L70)

So diffusion reinforces the same lesson: the issue is not one specific node size. The issue is persistence of split significance below the natural frontier.

**The Mathematical Rule**
The clean mathematical rule here is a scale-separation rule.

A healthy decomposition should satisfy:

$$
p_{\mathrm{sibling}}(\text{coarse scale}) \ll \alpha
\quad\text{and}\quad
p_{\mathrm{sibling}}(\text{finer scale}) \gtrsim \alpha
$$

In words:
- coarse splits should be strongly significant
- once you reach the natural cluster scale, significance should die out quickly

Single linkage approximately satisfies that. Average linkage and diffusion do not.

A more operational version is this:

Define a frontier just below the intended cluster scale, and let

$$
\pi_{\mathrm{fine}}(m)
=
\frac{\#\{u:\ |C_u|\le m,\ \text{sibling split accepted at }u\}}
{\#\{u:\ |C_u|\le m,\ \text{sibling test evaluated at }u\}}.
$$

Then:
- small $\pi_{\mathrm{fine}}(m)$ means the tree is stabilizing
- large $\pi_{\mathrm{fine}}(m)$ means the tree is still fragmenting and over-splitting is likely

Empirically in this case:
- average linkage has near-total acceptance for most tested nodes of size 3 to 7
- single linkage has almost none
- diffusion still has persistent acceptance on one branch below the 10-leaf level

So the rule we should learn is:

A good tree is not one that merely gets the root right. A good tree is one where sibling significance is strong at coarse scales and then collapses at fine scales. If significance persists for consecutive generations below the natural frontier, the decomposition will cascade into over-splitting.

That is the failure mode here.

If you want, I can next turn this into one of two concrete outputs:
1. a proposed stopping diagnostic you can compute automatically from the audit CSVs
2. a hypothesis for how to modify tree construction or Gate 3 so this failure mode is suppressed