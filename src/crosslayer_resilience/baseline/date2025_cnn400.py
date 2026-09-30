from __future__ import annotations

import torch
from torch import nn


class HistoricalIMUNormalizer(nn.Module):
    """
    Reproduce the recovered historical Protechto IMU normalization.

    Stored input:
        [batch, 40, 9]

    Stored channels:
        accelerometer XYZ
        gyroscope XYZ
        Euler-angle XYZ

    Effective task-model input:
        [batch, 40, 6]

    Euler-angle channels are retained in the historical stored-window
    contract but are not passed to the CNN.

    Provenance
    ----------
    Adapted from:
        muhammadtoqeerali/RC-RGD-IMU
        commit e1db7880e17a5632bc4ce238125923f986e85519
        src/imu_reliability/baseline/date2025_cnn400.py

    Historical source raw SHA-256:
        def71b3cebc0649c0d909e4ffd5dc04f177fcb795ff6ebfd13f90e214c67581d

    The normalization arithmetic is intentionally preserved.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,T,C], got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != 9:
            raise ValueError(
                f"Expected 9 stored channels, got {x.shape[-1]}"
            )

        acc_data, gyro_data, _angle_data = torch.split(
            x,
            3,
            dim=2,
        )

        # Preserve the two historical gyro divisions separately.
        gyro_data = gyro_data / torch.tensor(
            1000,
            dtype=torch.float32,
            device=x.device,
        )

        acc_data = acc_data / torch.tensor(
            [4000, 4000, 4000],
            dtype=torch.float32,
            device=x.device,
        )

        gyro_data = gyro_data / torch.tensor(
            [1800, 1800, 1800],
            dtype=torch.float32,
            device=x.device,
        )

        return torch.concat(
            [acc_data, gyro_data],
            dim=-1,
        )


class Date2025CNN400(nn.Module):
    """
    Reconstructed DATE-2025 400-ms pre-impact fall-detection CNN.

    Historical tensor geometry
    --------------------------
    Stored input        : 40 x 9
    Effective channels  : 6
    Convolution channels: 32
    Kernel size         : 4
    Pool size           : 2
    Flatten dimension   : 224
    Hidden FC           : 256
    Classes             : 2
    Parameters          : 63,173

    This module is the protected task workload only.

    It deliberately contains no:
    - sensor-integrity monitor
    - OOD mechanism
    - compute-integrity monitor
    - reliability fusion
    - recovery policy
    - cross-layer supervisor
    """

    WINDOW_SAMPLES = 40
    STORED_CHANNELS = 9
    EFFECTIVE_CHANNELS = 6
    NUM_CLASSES = 2
    EXPECTED_PARAMETER_COUNT = 63173

    def __init__(self) -> None:
        super().__init__()

        self.normalizer = HistoricalIMUNormalizer()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                self.EFFECTIVE_CHANNELS,
                32,
                4,
            ),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(
                32,
                32,
                4,
            ),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.4),
        )

        # 40 -> 37 -> 18 -> 15 -> 7
        # 7 * 32 = 224
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224, 256),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.NUM_CLASSES),
        )

    def _backbone(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,T,C], got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.WINDOW_SAMPLES:
            raise ValueError(
                "This protected model requires exactly "
                f"{self.WINDOW_SAMPLES} samples; "
                f"got {x.shape[1]}."
            )

        if x.shape[2] != self.STORED_CHANNELS:
            raise ValueError(
                "This protected model requires exactly "
                f"{self.STORED_CHANNELS} stored channels; "
                f"got {x.shape[2]}."
            )

        x = self.normalizer(x)
        x = x.permute(0, 2, 1)

        x = self.conv_1(x)
        x = self.conv_2(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._backbone(x)
        return self.fc(x)

    def forward_with_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return logits and the post-PReLU 256-D penultimate representation.

        This remains one task-model forward.
        """
        x = self._backbone(x)

        flat = self.fc[0](x)
        hidden = self.fc[1](flat)
        feature = self.fc[2](hidden)
        dropped = self.fc[3](feature)
        logits = self.fc[4](dropped)

        return logits, feature

    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
        )
