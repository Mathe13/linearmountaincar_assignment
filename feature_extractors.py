"""
Feature extractors (student assignment placeholders).

This file contains lightweight, instructor-friendly skeletons for feature
extractors. Students should implement the missing functionality as part of
the assignment. The goal is to provide clear method signatures, small
helper methods, and inline hints, but not a full implementation.

HINTS:
- Keep the public API (class names and method names) stable so tests and
  other modules can import these types.
- Implementations should work for the MountainCar 2D state: [position, velocity].
  Use `env.observation_space.low` and `.high` to obtain bounds.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import gymnasium as gym


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Students should implement `num_features` and `extract_features`.
    A minimal `normalize_state` helper is provided and may be used.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        # Derived classes should set/compute their own attributes
        # (for example, centers for RBF).
        self.n_features = self.num_features()

        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        self.state_bounds = list(zip(obs_low, obs_high))
        self.position_bounds = (obs_low[0], obs_high[0])
        self.velocity_bounds = (obs_low[1], obs_high[1])

    @abstractmethod
    def num_features(self) -> int:
        """Return the number of features in the feature vector."""
        pass

    @abstractmethod
    def extract_features(self, state: np.ndarray) -> np.ndarray:
        """Return feature vector for a given state.

        Args:
            state: raw environment state (e.g., [position, velocity])
        """
        pass

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize a state into [0, 1] for each dimension.

        This helper clamps out-of-bounds values to the [0,1] interval which
        is useful for grid-based or RBF centers defined in normalized space.
        """
        state = np.asarray(state, dtype=np.float64)
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        norm = (state - obs_low) / (obs_high - obs_low)
        return np.clip(norm, 0.0, 1.0)


class RBFFeatureExtractor(FeatureExtractor):
    """Skeleton RBF feature extractor for students to implement.

    Required tasks for students:
    - Implement `_create_rbf_centers` to place `n_centers` centers in normalized
      state space (0..1). A grid layout is a good starting point.
    - Implement `num_features` to return the number of centers.
    - Implement `extract_features` to compute Gaussian RBFs: exp(-||x-c||^2/(2*sigma^2)).

    HINTS:
    - Use `self.normalize_state(state)` to map raw states into [0,1].
    - If `n_centers` is not a perfect square, create a grid with at least
      ceil(sqrt(n_centers)) per dimension and then slice to the desired count.
    - Ensure the returned feature vector has length `self.n_features`.
    """

    def __init__(self, env: gym.Env, n_centers: int = 25, sigma: float = 0.1):
        # Student-implemented: store parameters and create centers
        self.n_centers = int(n_centers)
        self.sigma = float(sigma)

        self._create_rbf_centers()

        super().__init__(env)

    def num_features(self) -> int:
        # Students should return the number of RBF centers here.
        return int(self.n_centers)

    def _create_rbf_centers(self) -> None:
        """(Student) Create RBF centers in normalized state space.

        Students: implement this method to populate `self.centers`.
        The centers should be numpy array of shape (n_centers, 2) evenly distributed
        within the normalized [0,1] coordinates. The first center must be at (0,0) and the last at (1,1).

        """
        k = int(np.ceil(np.sqrt(self.n_centers)))
        if k <= 1:
            self.centers = np.zeros((self.n_centers, 2), dtype=np.float64)
            if self.n_centers > 0:
                self.centers[-1] = [1.0, 1.0]
        else:
            xs = np.linspace(0.0, 1.0, k)
            ys = np.linspace(0.0, 1.0, k)
            gx, gy = np.meshgrid(xs, ys)
            all_centers = np.column_stack([gx.ravel(), gy.ravel()])
            if len(all_centers) == self.n_centers:
                self.centers = all_centers
            else:
                self.centers = all_centers[:self.n_centers].copy()
                self.centers[-1] = [1.0, 1.0]

    def extract_features(self, state: np.ndarray) -> np.ndarray:
        """(Student) Compute RBF feature activations for a state.

        Returns a 1D numpy array of length `self.n_features`.
        """
        norm_state = self.normalize_state(state)
        diff = self.centers - norm_state
        sq_dist = np.sum(diff ** 2, axis=1)
        return np.exp(-sq_dist / (2.0 * (self.sigma ** 2)))