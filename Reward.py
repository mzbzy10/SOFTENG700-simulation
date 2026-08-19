"""The reward function and every knob that shapes it.

Swap the body of compute_reward() to change the reward algorithm; the Simulator
only calls this one entry point.
"""

import numpy as np

from Environments import NORM_MAX_QUEUE, NORM_MAX_DEADLINE


# Slice-specific weights. All four inputs below are normalized to 0-1, so these
# weights are directly comparable to each other.
W_THROUGHPUT = np.array([1.0, 0.5, 0.3])  # eMBB cares most about throughput
W_DEADLINE   = np.array([0.5, 1.5, 0.2])  # URLLC deadline misses hurt most, but not
                                          # so much that other slices stop mattering
W_QUEUE      = np.array([0.2, 0.8, 0.1])  # URLLC queue backlog is bad
W_STARVE     = 1.0                        # fairness: same weight for every slice

# Starvation streaks are clipped at this multiple of the slice's own deadline.
# A 1x cap made the penalty flat for most of a long episode, giving the agent no
# incentive to recover once starved.
STARVE_CLIP = 3.0

# A slice counts as neglected on a step if it had demand but was served less
# than this fraction of it. Must be a service *ratio*, not "alloc == 0": the DQN
# action space only enumerates splits with every slice >= 5 PRBs, so a literal
# zero-allocation test can never fire. Used by Simulator to advance the streak.
STARVE_SERVICE_THRESHOLD = 0.1


def compute_reward(served, demand, queue, alloc, deadline_miss, starve_steps, total_prb):
    """Scalar reward for one simulation step.

    All arguments are length-3 arrays over (eMBB, URLLC, mMTC) except total_prb.

    served         PRBs actually delivered to each slice this step
    demand         aggregate queued work per slice, before serving
    queue          number of incomplete tasks per slice, after serving
    alloc          PRBs allocated to each slice by the allocator
    deadline_miss  fraction of queued tasks past deadline, per slice (0-1)
    starve_steps   consecutive steps each slice has been neglected
    total_prb      the allocator's total PRB budget
    """
    # fraction of demand actually served per slice (0-1)
    throughput_rate = np.divide(served, demand, out=np.zeros(3), where=demand > 0)

    # queue backlog against a fixed cap
    norm_queue = np.clip(queue / NORM_MAX_QUEUE, 0.0, 1.0)

    # how long a slice has been left with demand but under-served, relative to
    # its own deadline. Squared so brief neglect is cheap but sustained
    # abandonment is punished heavily.
    norm_starve = np.clip(starve_steps / NORM_MAX_DEADLINE, 0.0, STARVE_CLIP)

    # NOTE: a `+ 0.5 * utilisation` term used to sit here. Every action in every
    # allocator sums to exactly total_prb, so utilisation was always 1.0 and the
    # term was a constant offset contributing nothing to the gradient. Removed.
    # `alloc` and `total_prb` stay in the signature so a reward that does shape
    # allocation directly can be dropped in without touching the Simulator.

    return (
        (W_THROUGHPUT * throughput_rate).sum()
        - (W_DEADLINE * deadline_miss).sum()
        - (W_QUEUE    * norm_queue).sum()
        - W_STARVE * (norm_starve ** 2).sum()
    )
