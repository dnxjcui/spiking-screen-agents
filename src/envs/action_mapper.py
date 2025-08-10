from __future__ import annotations


class ActionMapper:
    """Maps reduced action space (NOOP, UP, DOWN) -> environment action indices dynamically."""

    def __init__(self, env):
        meanings = env.unwrapped.get_action_meanings()
        self.idx_noop = None
        self.idx_up = None
        self.idx_down = None
        for i, m in enumerate(meanings):
            if "NOOP" in m and self.idx_noop is None:
                self.idx_noop = i
            if "UP" in m and self.idx_up is None:
                self.idx_up = i
            if "DOWN" in m and self.idx_down is None:
                self.idx_down = i
        if self.idx_noop is None:
            self.idx_noop = 0
        if self.idx_up is None:
            self.idx_up = 2
        if self.idx_down is None:
            self.idx_down = 3
        self.action_list = [self.idx_noop, self.idx_up, self.idx_down]

    def to_env(self, policy_action: int) -> int:
        return self.action_list[int(policy_action)]

    def n_actions(self) -> int:
        return 3

