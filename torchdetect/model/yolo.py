import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule


class YOLOv1(LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = False,
        num_grids: int = 7,
        num_boxes: int = 2,
        lmbda_coord: float = 5,
        lmbda_noobj: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_grids = num_grids
        self.num_boxes = num_boxes
        self.lmbda_coord = lmbda_coord
        self.lmbda_noobj = lmbda_noobj
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = models.__dict__[self.backbone](pretrained=self.pretrained)
            out_features = self._num_grids ** 2 * (
                self.num_boxes * 5 + self.num_classes
            )
            self._model.fc = nn.Linear(in_features=2048, out_features=out_features)
        return self._model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        pred = self(images)
        loss = self.loss(
            pred.view(
                -1,
                self.num_grids,
                self.num_grids,
                self.num_boxes * 5 + self.num_classes,
            ),
            target,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ...

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the YOLOv1 loss.

        Arguments:
            pred: an NxSxSx(Bx5+C) tensor where N is the number of images, S is the
                  number of grids, B is the number of boxes for grid, and C is the
                  number of classes.
            target: an NxSxSx(Bx5) tensor where N is the number of images, S is the
                    number of grids, B is the number of boxes for grid, and C is the
                    number of classes.
        """
