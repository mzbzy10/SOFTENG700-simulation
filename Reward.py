"""The reward function and every knob that shapes it.

Swap the body of compute_reward() to change the reward algorithm; the Simulator
only calls this one entry point.

Design rationale (Zangooei et al., "RL for RRM in RAN Slicing: A Survey", 2022,
Section III-A "Reward"):

  - A weighted sum of throughput/utilisation against an SLA term is the [4]
    formulation, which [5] rejects: a linear blend "could lead to a blind
    sacrifice of SLA in exchange for an increase in SE". The previous reward
    here had exactly that shape, so it was replaced.
  - The survey's preferred single-objective form is SSR x SE ([9]). SE is not
    computable in this simulator: every action sums to total_prb and serving is
    1:1 PRBs-to-work with no channel/MCS model, so SE would be a constant and
    the product collapses back to SSR.
  - What remains is SLA satisfaction as the objective ([6]), which sits on the
    safe side of every criticism the survey makes and aligns the reward with
    SSR, the metric used for evaluation.
  - SSR itself is measured on a *per-slice* KPI (rate / delay / buffer) rather
    than one deadline metric with three thresholds — see Environments.SLA.
"""

import numpy as np

from Environments import NORM_MAX_QUEUE


# Weight on the backlog shaping term.
#
# This term is not a modelling choice, it is credit assignment: mMTC's deadline
# is 100 steps, so an SSR-only reward gives almost no gradient on mMTC within an
# episode while URLLC (deadline 10) dominates the signal. Normalized queue
# length supplies a dense proxy. Keep it small relative to the SSR term, whose
# range is 0-3.
LAMBDA_QUEUE = 0.1

# A slice counts as neglected on a step if it had demand but was served less
# than this fraction of it. Must be a service *ratio*, not "alloc == 0": the DQN
# action space only enumerates splits with every slice >= 5 PRBs, so a literal
# zero-allocation test can never fire.
#
# DIAGNOSTIC ONLY as of the SSR reward — Simulator still tracks and plots the
# starvation streak, but it no longer enters the reward. A starved slice now
# loses SSR directly, so the explicit penalty term (and its hand-tuned exponent
# and clip, which had no basis in the literature) is redundant.
STARVE_SERVICE_THRESHOLD = 0.1


def compute_reward(served, demand, queue, alloc, ssr, starve_steps, total_prb):
    """Scalar reward for one simulation step.

        R(t) = sum_i SSR_i(t)  -  LAMBDA_QUEUE * sum_i q_hat_i(t)

    where, per slice i:
        SSR_i   = satisfaction of slice i's own KPI, from Simulator.get_ssr:
                  eMBB min data rate, URLLC max delay, mMTC max buffer   (0-1)
        q_hat_i = queue_i / NORM_MAX_QUEUE   normalized backlog          (0-1)

    Slices are weighted equally on purpose. Any other weighting is a claim about
    relative slice priority that would have to be justified separately, and
    per-slice SLA difficulty is already expressed by the deadlines (10 / 80 /
    100 steps) rather than needing to be encoded a second time.

    Range: [-3 * LAMBDA_QUEUE, 3].

    All arguments are length-3 arrays over (eMBB, URLLC, mMTC) except total_prb.
    Several are unused by this particular reward but stay in the signature so an
    alternative algorithm can be dropped in without touching the Simulator.

    served         PRBs actually delivered to each slice this step      (unused)
    demand         aggregate queued work per slice, before serving      (unused)
    queue          number of incomplete tasks per slice, after serving
    alloc          PRBs allocated to each slice by the allocator        (unused)
    ssr            per-slice SLA satisfaction ratio, each on its own KPI (0-1)
    starve_steps   consecutive steps each slice has been neglected      (unused)
    total_prb      the allocator's total PRB budget                     (unused)
    """
    # per-slice backlog, against a fixed cap (see Environments.NORM_MAX_QUEUE)
    norm_queue = np.clip(queue / NORM_MAX_QUEUE, 0.0, 1.0)

    return np.asarray(ssr).sum() - LAMBDA_QUEUE * norm_queue.sum()
