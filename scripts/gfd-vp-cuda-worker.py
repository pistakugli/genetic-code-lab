#!/usr/bin/env python3
import json
import sys
import time

import torch


if not torch.cuda.is_available():
    raise RuntimeError("GFD-VP CUDA worker requires an available CUDA device")

DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
torch.cuda.init()


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def fit(request):
    started = time.perf_counter()
    x_cpu = torch.tensor(request["inputs"], dtype=DTYPE, pin_memory=True)
    y_cpu = torch.tensor(request["outputs"], dtype=DTYPE, pin_memory=True)
    x = x_cpu.to(DEVICE, non_blocking=True)
    y = y_cpu.to(DEVICE, non_blocking=True)
    sample_count, dimension = x.shape
    term_count = request["termCount"]
    restart_count = request["restarts"]
    batch_size = min(request["batchSize"], sample_count)
    steps = request["optimizationSteps"]
    profile_head = request["profileOutputLayer"]
    validation_count = max(1, int(sample_count * 0.2)) if sample_count > 1 else sample_count
    training_count = sample_count - validation_count if sample_count > 1 else sample_count
    x_train, x_validation = x[:training_count], x[training_count:]
    y_train, y_validation = y[:training_count], y[training_count:]
    if training_count < dimension + term_count + 1:
        raise ValueError("GFD-VP received fewer training samples than affine features")

    seed = int(request["randomSeed"]) % (2**63 - 1)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    normals = torch.randn(
        restart_count, term_count, dimension, device=DEVICE, dtype=DTYPE,
        generator=generator, requires_grad=True
    )
    with torch.no_grad():
        normals /= normals.norm(dim=2, keepdim=True).clamp_min(1e-10)
    normals.requires_grad_(True)
    offsets = (torch.rand(
        restart_count, term_count, device=DEVICE, dtype=DTYPE, generator=generator
    ) * 0.7 - 0.35).requires_grad_(True)
    base = torch.zeros(restart_count, dimension + 1, device=DEVICE, dtype=DTYPE)
    readouts = torch.zeros(restart_count, term_count, device=DEVICE, dtype=DTYPE)
    if restart_count > 1:
        readouts.copy_(torch.randn(
            restart_count, term_count, device=DEVICE, dtype=DTYPE, generator=generator
        ) * 0.2)

    hidden_optimizer = torch.optim.Adam([normals, offsets], lr=0.015)
    head_parameters = torch.nn.Parameter(torch.cat([base, readouts], dim=1))
    joint_optimizer = torch.optim.Adam([normals, offsets, head_parameters], lr=0.02)

    def hidden_features(inputs):
        preactivation = torch.einsum("rhd,nd->rhn", normals, inputs) + offsets[:, :, None]
        return torch.cat([
            torch.ones(restart_count, inputs.shape[0], 1, device=DEVICE, dtype=DTYPE),
            inputs[None, :, :].expand(restart_count, -1, -1),
            torch.relu(preactivation).transpose(1, 2)
        ], dim=2)

    def profile_output(inputs, targets):
        with torch.no_grad():
            design = hidden_features(inputs)
            gram = torch.einsum("rnp,rnq->rpq", design, design)
            rhs = torch.einsum("rnp,n->rp", design, targets)
            scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0)
            identity = torch.eye(gram.shape[1], device=DEVICE, dtype=DTYPE)
            gram = gram + (1e-8 * scale)[:, None, None] * identity[None, :, :]
            head_parameters.copy_(torch.linalg.solve(gram, rhs[:, :, None]).squeeze(2))

    def predict_batch(inputs):
        design = hidden_features(inputs)
        return torch.einsum("rnp,rp->rn", design, head_parameters)

    refit_count = max(256, batch_size * 2)
    for step in range(steps):
        if profile_head and step % 8 == 0:
            indices = torch.randint(training_count, (min(refit_count, training_count),),
                                    device=DEVICE, generator=generator)
            profile_output(x_train[indices], y_train[indices])
        indices = torch.randint(training_count, (batch_size,), device=DEVICE, generator=generator)
        xb, yb = x_train[indices], y_train[indices]
        if profile_head:
            hidden_optimizer.zero_grad(set_to_none=True)
            loss = (predict_batch(xb) - yb[None, :]).square().mean(dim=1).sum()
            loss.backward()
            hidden_optimizer.step()
        else:
            joint_optimizer.zero_grad(set_to_none=True)
            loss = (predict_batch(xb) - yb[None, :]).square().mean(dim=1).sum()
            loss.backward()
            joint_optimizer.step()
        with torch.no_grad():
            lengths = normals.norm(dim=2, keepdim=True).clamp_min(1e-10)
            head_parameters[:, dimension + 1:].mul_(lengths.squeeze(2))
            normals.div_(lengths)
            offsets.div_(lengths.squeeze(2))

    if profile_head:
        profile_output(x_train, y_train)
    with torch.no_grad():
        train_predictions = predict_batch(x_train)
        validation_predictions = predict_batch(x_validation)
        training_maes = (train_predictions - y_train[None, :]).abs().mean(dim=1)
        validation_maes = (validation_predictions - y_validation[None, :]).abs().mean(dim=1)
        best_index = int(validation_maes.argmin().item())
        best_training_mae = float(training_maes[best_index].item())
        best_validation_mae = float(validation_maes[best_index].item())
        best_normals = normals[best_index].detach().cpu().tolist()
        best_offsets = offsets[best_index].detach().cpu().tolist()
        best_head = head_parameters[best_index].detach().cpu().tolist()
        terms = []
        for index in range(term_count):
            terms.append({
                "normal": best_normals[index],
                "offset": best_offsets[index],
                "readout": best_head[dimension + 1 + index]
            })
        cuda_memory = torch.cuda.max_memory_allocated(DEVICE)
        torch.cuda.synchronize(DEVICE)

    return {
        "dimension": dimension,
        "baseIntercept": best_head[0],
        "baseWeights": best_head[1:dimension + 1],
        "terms": terms,
        "diagnostics": {
            "totalOracleCalls": sample_count,
            "sampleCount": sample_count,
            "trainingSamples": training_count,
            "validationSamples": validation_count,
            "restartsAttempted": restart_count,
            "selectedRestart": best_index,
            "trainingMae": best_training_mae,
            "validationMae": best_validation_mae,
            "reconstructionComplete": best_validation_mae <= 1e-5,
            "computeDevice": str(normals.device),
            "cudaMemoryAllocatedBytes": int(cuda_memory),
            "trainingTimeMs": (time.perf_counter() - started) * 1000.0,
            "dtype": str(normals.dtype)
        },
        "reconstructionComplete": best_validation_mae <= 1e-5
    }


torch.cuda.reset_peak_memory_stats(DEVICE)
emit({
    "ready": True,
    "computeDevice": str(DEVICE),
    "deviceName": torch.cuda.get_device_name(DEVICE),
    "torchVersion": torch.__version__,
    "cudaVersion": torch.version.cuda
})
for line in sys.stdin:
    request = json.loads(line)
    if request.get("action") == "close":
        break
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        emit({"ok": True, "model": fit(request)})
    except Exception as error:
        emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
