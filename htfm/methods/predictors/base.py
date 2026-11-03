"""Abstract Predictor base class.

A Predictor encapsulates the prediction parameterization the model is
trained under. Two questions per parameterization:

  1. Training:  what is the regression target given (X1, X0, G)?
  2. Sampling:  given the model output, how do we recover the velocity v?

Currently implemented in subclasses:
  - VelocityPredictor: model directly outputs v(t, X_t).
  - EpsilonPredictor:  model outputs the standardized Gaussian noise eps;
                       v ~ barsigma_T * eps under the straight-line schedule.
"""


class Predictor:
    """Abstract base; subclasses override compute_target / to_velocity."""

    def compute_target(self, X1, x0_ext, G):
        """Training-side: return the tensor the model's raw output is regressed against.

        Args:
            X1:     [B*total_mc, ...] noise endpoint.
            x0_ext: [B*total_mc, ...] data points (repeated to MC layout).
            G:      [B*total_mc, ...] standard-normal noise that produced X1.

        Returns:
            target tensor of the same shape as the model's output.
        """
        raise NotImplementedError

    def to_velocity(self, model_output, x_t, t):
        """Sampling-side: map model raw output to velocity v(t, x_t).

        Args:
            model_output: [B, ...] raw output of the model.
            x_t:          [B, ...] current state along the ODE trajectory.
            t:            [B] timestep tensor in [0, 1].

        Returns:
            velocity tensor of the same shape as x_t.
        """
        raise NotImplementedError
