import ast
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, default_collate

import star_classifier as classifier

torch.set_num_threads(1)


def frame(count=65):
    rng = np.random.default_rng(12)
    return pd.DataFrame({"a": rng.normal(size=count), "b": rng.normal(size=count), "class": np.arange(count) % 3})


@pytest.mark.parametrize("count", [1, 63, 64, 65, 129])
def test_test_loader_covers_every_sample(count):
    dataset = classifier.CustomDataset(frame(count))
    _, loader = classifier.make_loaders(dataset, dataset, batch_size=64)
    assert sum(len(batch["label"]) for batch in loader) == count
    result = classifier.evaluate(nn.Linear(2, 3), loader)
    assert result["samples"] == count
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_unlabelled_dataset_and_prediction_work():
    dataset = classifier.CustomDataset(frame().drop(columns=["class"]), is_train=False)
    assert set(dataset[0]) == {"feature"}
    result = classifier.predict(classifier.Net(2, 3), DataLoader(dataset, batch_size=64))
    assert result.shape == (65, 3)
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)


class RepeatingBatchSampler(BatchSampler):
    def __iter__(self):
        yield [0] * len(self.sampler)


def repeated_collate(batch):
    return default_collate([batch[0]] * len(batch))


@pytest.mark.parametrize("function", [classifier.evaluate, classifier.predict])
@pytest.mark.parametrize("kind", ["repeated", "reverse", "batch_override", "collate_override", "unordered_delivery"])
def test_evaluation_requires_once_in_dataset_order(function, kind):
    dataset = classifier.CustomDataset(frame(3))
    if kind == "batch_override":
        loader = DataLoader(dataset, batch_sampler=RepeatingBatchSampler(SequentialSampler(dataset), 3, False))
    elif kind == "collate_override":
        loader = DataLoader(dataset, batch_size=3, collate_fn=repeated_collate)
    elif kind == "unordered_delivery":
        loader = DataLoader(dataset, batch_size=3)
        loader.in_order = False
    else:
        loader = DataLoader(dataset, batch_size=3, sampler=[0, 0, 0] if kind == "repeated" else [2, 1, 0])
    with pytest.raises(ValueError, match="sequential"):
        function(nn.Linear(2, 3), loader)


def test_predictions_preserve_dataset_row_order_across_batches():
    values = pd.DataFrame({"a": [4., -4., 1.], "b": [0., 0., 0.]})
    dataset = classifier.CustomDataset(values, is_train=False)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    actual = classifier.predict(model, DataLoader(dataset, batch_size=2))
    expected = torch.tensor(values.to_numpy(), dtype=torch.float32).softmax(1).numpy()
    np.testing.assert_allclose(actual, expected)


def test_unlabelled_inference_refuses_target_column():
    with pytest.raises(ValueError, match="target column"):
        classifier.CustomDataset(frame(), is_train=False)


def test_evaluation_loss_is_sample_weighted():
    data = pd.DataFrame({"a": [8.] * 64 + [-8.], "b": [0.] * 65, "class": [0] * 65})
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    dataset = classifier.CustomDataset(data)
    result = classifier.evaluate(model, DataLoader(dataset, batch_size=64))
    expected = nn.functional.cross_entropy(torch.tensor(data[["a", "b"]].values, dtype=torch.float32), torch.zeros(65, dtype=torch.long)).item()
    assert result["loss"] == pytest.approx(expected, abs=1e-6)
    assert result["samples"] == 65


def test_missing_class_auc_is_undefined_not_zero():
    result = classifier.classification_metrics([0, 0], [[.8, .1, .1], [.8, .1, .1]], 0.2)
    assert result["auc_roc"] is None
    assert result["sensitivity"] == [1., None, None]
    assert result["specificity"][0] is None
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_binary_auc_uses_positive_class_probability():
    result = classifier.classification_metrics([0, 1], [[.9, .1], [.2, .8]], 0.2)
    assert result["auc_roc"] == 1


def test_multiclass_auc_is_correct():
    result = classifier.classification_metrics([0, 1, 2], np.eye(3), 0)
    assert result["auc_roc"] == 1


@pytest.mark.parametrize("function", [classifier.evaluate, classifier.predict])
def test_incomplete_evaluation_loader_is_rejected(function):
    loader = DataLoader(classifier.CustomDataset(frame(63)), batch_size=64, drop_last=True)
    with pytest.raises(ValueError, match="drop"):
        function(nn.Linear(2, 3), loader)


@pytest.mark.parametrize("function", [classifier.evaluate, classifier.predict])
def test_empty_evaluation_loader_is_rejected(function):
    with pytest.raises(ValueError, match="nonempty"):
        function(nn.Linear(2, 3), DataLoader([]))


def test_train_epoch_updates_real_lif_network():
    classifier.set_seed(42)
    model = classifier.Net(2, 3)
    optimizer = torch.optim.Adam(model.parameters())
    before = model.fc3.bias.detach().clone()
    loss = classifier.train_epoch(model, DataLoader(classifier.CustomDataset(frame()), batch_size=16), optimizer)
    assert np.isfinite(loss)
    # The output bias must train even when no hidden neuron fires initially.
    assert not torch.equal(model.fc3.bias.detach(), before)
    assert model(torch.ones(5, 2)).shape == (5, 3)


def test_single_step_network_resets_state_per_call():
    classifier.set_seed(7)
    model = classifier.Net(2, 3).eval()
    data = torch.ones(4, 2)
    with torch.no_grad():
        first, second = model(data), model(data)
    assert torch.equal(first, second)


def test_device_mismatch_is_explicit():
    with pytest.raises(ValueError, match="Move the model"):
        classifier.evaluate(nn.Linear(2, 3), DataLoader(classifier.CustomDataset(frame())), device="meta")


def test_generic_accelerator_name_preserves_the_models_actual_index():
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:1"))]))
    assert classifier._check_device(model, "cuda") == torch.device("cuda:1")
    with pytest.raises(ValueError):
        classifier._check_device(model, "cuda:0")


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_real_accelerator_training_and_evaluation_when_available(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip("No " + device + " runtime in this test environment")
    model = classifier.Net(2, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loader = DataLoader(classifier.CustomDataset(frame(65)), batch_size=32)
    assert np.isfinite(classifier.train_epoch(model, loader, optimizer, device=device))
    assert classifier.evaluate(model, loader, device=device)["samples"] == 65


def test_preprocessing_fits_training_rows_only():
    original = frame(60)
    training, testing = train_test_split(original, test_size=.2, random_state=42, stratify=original["class"])
    train, test, scaler, encoder = classifier.prepare_split(original, oversample=False)
    np.testing.assert_allclose(scaler.mean_, training[["a", "b"]].mean().to_numpy())
    np.testing.assert_allclose(test.features, scaler.transform(testing[["a", "b"]]), rtol=1e-6, atol=1e-6)
    assert len(train) == 48 and len(test) == 12


def test_smote_does_not_modify_test_examples():
    original = frame(60)
    _, plain, _, _ = classifier.prepare_split(original, oversample=False)
    _, oversampled, _, _ = classifier.prepare_split(original, oversample=True)
    np.testing.assert_array_equal(plain.features, oversampled.features)
    np.testing.assert_array_equal(plain.labels, oversampled.labels)


def test_test_set_is_evaluated_once_per_model(monkeypatch):
    original = classifier.evaluate
    counts = []
    def observe(model, loader, **kwargs):
        counts.append(len(loader.dataset))
        return original(model, loader, **kwargs)
    monkeypatch.setattr(classifier, "evaluate", observe)
    result = classifier.run_experiment(frame(60), epochs=2)
    assert counts == [12, 12]
    assert all(len(history) == 2 for history in result["history"].values())


def test_same_seed_reproduces_cpu_results():
    first = classifier.run_experiment(frame(60), epochs=1)
    second = classifier.run_experiment(frame(60), epochs=1)
    assert first["history"] == second["history"]
    assert first["metrics"] == second["metrics"]


@pytest.mark.parametrize("bad", [pd.DataFrame({"a": [np.nan], "class": [0]}),
                                pd.DataFrame({"a": [1.], "class": [0.5]}),
                                pd.DataFrame({"a": [1.], "class": [-1]})])
def test_bad_data_is_rejected(bad):
    with pytest.raises(ValueError):
        classifier.CustomDataset(bad)


def test_notebook_is_output_free_valid_python_and_runs_on_synthetic_csv(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "SNN_Star_classification.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert set(notebook.metadata) == {"kernelspec", "language_info"}
    data = tmp_path / "data"
    data.mkdir()
    frame(60).to_csv(data / "star_classification.csv", index=False)
    monkeypatch.chdir(tmp_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {}
    for cell in notebook.cells:
        assert not cell.metadata and "attachments" not in cell
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            ast.parse(cell.source)
            exec(cell.source, namespace)
    assert namespace["result"]["metrics"]["single_step_lif"]["samples"] == 12
    plt.close("all")
