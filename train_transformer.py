import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


# -----------------------------------
# DEVICE
# -----------------------------------

device = torch.device("cpu")


# -----------------------------------
# LOAD DATA
# -----------------------------------

X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")


# -----------------------------------
# SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# CLASS WEIGHTS
# -----------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

pos_weight = torch.tensor(
    1.2,
    dtype=torch.float32
).to(device)


# -----------------------------------
# TENSORS
# -----------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)


train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=256
)


# -----------------------------------
# POSITIONAL ENCODING
# -----------------------------------

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        hidden_dim,
        max_len=100
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            hidden_dim
        )

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                hidden_dim,
                2
            ) * (-np.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )


    def forward(self, x):

        return x + self.pe[:, :x.size(1)]



# -----------------------------------
# MODEL
# -----------------------------------

class StockTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.pos_encoder = PositionalEncoding(
            hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )


    def forward(self, x):

        x = self.embedding(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        # use LAST timestep
        #x = x[:, -1, :]

        x = torch.mean(
            x,
            dim=1
        )

        x = self.fc(x)

        return x


model = StockTransformer(
    input_dim=X.shape[2]
).to(device)


# -----------------------------------
# LOSS
# -----------------------------------

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------------
# TRAIN
# -----------------------------------

epochs = 15

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(
            batch_x
        ).squeeze()

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss:.4f}"
    )


# -----------------------------------
# EVALUATE
# -----------------------------------

model.eval()

all_preds = []
all_true = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)

        logits = model(
            batch_x
        ).squeeze()

        probs = torch.sigmoid(
            logits
        )

        threshold = 0.50
        preds = (probs > threshold).int()
        
        all_preds.extend(
            preds.cpu().numpy()
        )

        all_true.extend(
            batch_y.numpy()
        )


print("\n==================================================")
print("TRANSFORMER RESULTS")
print("==================================================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            all_true,
            all_preds
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            all_true,
            all_preds
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            all_true,
            all_preds
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            all_true,
            all_preds
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        all_true,
        all_preds
    )
)