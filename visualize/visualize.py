import matplotlib.pyplot as plt
import numpy as np


class Visualize:
    """Interactive visualizer for DQN outputs.""" 
    def __init__(self, action_dim: int, figsize: tuple=(8, 12)) -> None:
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")
        self.action_dim = action_dim

        plt.ion()

        fig, ax = plt.subplots(nrows=2, figsize=figsize)
        ax = ax.flatten()

        ax[0].set_xlabel("Actions")
        ax[0].set_ylabel("Action value $Q(. ,s_{t})$")
        ax[0].set_xticks(np.arange(action_dim))
        ax[0].set_xlim(-0.5, action_dim - 0.5)

        x = np.arange(action_dim, dtype=np.int64)
        heights = np.zeros(action_dim, dtype=np.float32)
        
        # Init bar plot
        self.bars = ax[0].bar(x, heights)
        
        # Init Q plot
        ax[1].set_xlabel("Timestep $(t)$")
        ax[1].set_ylabel("Action value $Q(s_{t}, a_{t})$")

        (self.line_q,) = ax[1].plot([], [])

        self.counter = 0
        self.steps = []
        self.q_values = []
        self.v_values = []

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

    def update(self, q_values: np.ndarray, q_value: float) -> None:
        """Update plots.""" 
        self.steps.append(self.counter)
        self.q_values.append(q_value)

        # Update bar plot
        for b, h in zip(self.bars, q_values):
            b.set_height(h) 
        ymin = min(0.0, float(q_values.min()))
        ymax = float(q_values.max())        
        if ymin == ymax:
            ymax = ymin + 1e-6 
        self.ax[0].set_ylim(ymin * 1.1, ymax * 1.1)

        # Update Q plot
        self.line_q.set_data(self.steps, self.q_values)
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.counter += 1
    
    def reset(self) -> None:
        """Reset plots.""" 
        self.counter = 0
        self.steps.clear()
        self.q_values.clear()

        self.line_q.set_data([], [])

        for b in self.bars:
            b.set_height(0.0)

        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()