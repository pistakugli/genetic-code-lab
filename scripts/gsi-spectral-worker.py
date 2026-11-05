#!/usr/bin/env python3
import json
import math
import sys
import time
import traceback

import torch


if not torch.cuda.is_available():
    raise RuntimeError("GSMI spectral worker requires an available CUDA device")

DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
torch.cuda.init()


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def solve(request):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    x_cpu = torch.tensor(request["inputs"], dtype=DTYPE, pin_memory=True)
    y_cpu = torch.tensor(request["outputs"], dtype=DTYPE, pin_memory=True)
    x = x_cpu.to(DEVICE, non_blocking=True)
    y = y_cpu.to(DEVICE, non_blocking=True)
    sample_count, dimension = x.shape
    sigma = float(request["sampleSigma"])
    sigma2 = sigma * sigma
    sigma4 = sigma2 * sigma2
    moment_order = request["momentOrder"]

    design = torch.cat([
        torch.ones(sample_count, 1, dtype=DTYPE, device=DEVICE), x
    ], dim=1)
    crossfit = torch.empty_like(y)
    first = torch.arange(0, sample_count, 2, device=DEVICE)
    second = torch.arange(1, sample_count, 2, device=DEVICE)
    first_fit = torch.linalg.lstsq(design[first], y[first, None]).solution[:, 0]
    second_fit = torch.linalg.lstsq(design[second], y[second, None]).solution[:, 0]
    crossfit[second] = y[second] - design[second] @ first_fit
    crossfit[first] = y[first] - design[first] @ second_fit

    hessian_scores = x[:, :, None] * x[:, None, :] / sigma4
    diagonal = torch.arange(dimension, device=DEVICE)
    hessian_scores[:, diagonal, diagonal] -= 1.0 / sigma2
    second_moment = torch.einsum("n,nij->ij", crossfit, hessian_scores) / sample_count
    second_moment = (second_moment + second_moment.T) * 0.5

    if moment_order == "second-and-third":
        third_moment = torch.zeros((dimension, dimension, dimension), dtype=DTYPE, device=DEVICE)
        chunk_size = max(8, min(128, 32768 // max(1, dimension * dimension)))
        for start in range(0, sample_count, chunk_size):
            end = min(sample_count, start + chunk_size)
            xb = x[start:end]
            yb = crossfit[start:end]
            cubic = torch.einsum("n,ni,nj,nk->ijk", yb, xb, xb, xb) / sigma4
            weighted_x = torch.einsum("n,nk->k", yb, xb) / sigma2
            cubic -= torch.einsum("ij,k->ijk", torch.eye(dimension, dtype=DTYPE, device=DEVICE), weighted_x)
            cubic -= torch.einsum("ik,j->ijk", torch.eye(dimension, dtype=DTYPE, device=DEVICE), weighted_x)
            cubic -= torch.einsum("jk,i->ijk", torch.eye(dimension, dtype=DTYPE, device=DEVICE), weighted_x)
            third_moment.add_(cubic / sample_count)
        third_moment = (
            third_moment
            + third_moment.permute(0, 2, 1)
            + third_moment.permute(1, 0, 2)
            + third_moment.permute(1, 2, 0)
            + third_moment.permute(2, 0, 1)
            + third_moment.permute(2, 1, 0)
        ) / 6.0
    else:
        third_moment = None

    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    maximum = float(eigenvalues.abs().max().item()) if dimension else 0.0
    second_contributions = crossfit[:, None, None] * hessian_scores
    block_count = min(8, sample_count)
    block_noise = 0.0
    for block in torch.tensor_split(second_contributions, block_count, dim=0):
        block_estimate = block.mean(dim=0)
        block_noise = max(
            block_noise,
            float(torch.linalg.matrix_norm(block_estimate - second_moment, ord=2).item())
        )
    rank_threshold = max(1e-10, 3.0 * block_noise / math.sqrt(block_count))
    keep = eigenvalues.abs() > rank_threshold
    estimated_rank = int(keep.sum().item())
    terms = []
    unresolved = 0
    pencil_residual = None

    if estimated_rank and third_moment is not None:
        basis = eigenvectors[:, keep]
        reduced_second = basis.T @ second_moment @ basis
        reduced_second = (reduced_second + reduced_second.T) * 0.5
        reduced_rank = reduced_second.shape[0]
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(request["randomSeed"]) % (2**63 - 1))
        probe = torch.randn(dimension, dtype=DTYPE, device=DEVICE, generator=generator)

        def reduced_pencil(direction):
            contracted = torch.einsum("ijk,k->ij", third_moment, direction)
            reduced_third = basis.T @ contracted @ basis
            return torch.linalg.solve(reduced_second, reduced_third)

        generic = reduced_pencil(probe)
        eigenvalues_generic, eigenvectors_generic = torch.linalg.eig(generic)
        inverse_eigenvectors = torch.linalg.inv(eigenvectors_generic)
        slopes = torch.empty((reduced_rank, dimension), dtype=torch.float64, device=DEVICE)
        for axis in range(dimension):
            direction = torch.zeros(dimension, dtype=DTYPE, device=DEVICE)
            direction[axis] = 1.0
            pencil = reduced_pencil(direction).to(torch.complex128)
            diagonalized = inverse_eigenvectors @ pencil @ eigenvectors_generic
            slopes[:, axis] = diagonalized.diagonal().real

        q_vectors = slopes
        recovered_normals = []
        recovered_offsets = []
        for index in range(reduced_rank):
            q = q_vectors[index]
            q_length = float(torch.linalg.vector_norm(q).item())
            if not math.isfinite(q_length) or q_length <= 1e-8 * max(1.0, sigma):
                unresolved += 1
                continue
            normal = q / q_length
            offset = -q_length
            recovered_normals.append(normal)
            recovered_offsets.append(offset)

        if recovered_normals:
            normal_matrix = torch.stack(recovered_normals, dim=1)
            atoms = torch.einsum("ir,jr->ijr", normal_matrix, normal_matrix).reshape(-1, len(recovered_normals))
            weights = torch.linalg.lstsq(atoms, second_moment.reshape(-1, 1)).solution[:, 0]
            normal_density = torch.exp(-torch.tensor(recovered_offsets, dtype=DTYPE, device=DEVICE).square()
                                       / (2 * sigma2)) / (math.sqrt(2 * math.pi) * sigma)
            readouts = weights / normal_density.clamp_min(torch.finfo(DTYPE).tiny)
            terms = [{
                "normal": normal_matrix[:, index].detach().cpu().tolist(),
                "offset": float(recovered_offsets[index]),
                "readout": float(readouts[index].item())
            } for index in range(len(recovered_normals))]

        second_reconstruction = torch.zeros_like(second_moment)
        third_reconstruction = torch.zeros_like(third_moment)
        for term in terms:
            normal = torch.tensor(term["normal"], dtype=DTYPE, device=DEVICE)
            weight = term["readout"] * math.exp(-term["offset"] ** 2 / (2 * sigma2)) / (math.sqrt(2 * math.pi) * sigma)
            second_reconstruction += weight * torch.outer(normal, normal)
            third_reconstruction += (
                -term["offset"] * weight
                * torch.einsum("i,j,k->ijk", normal, normal, normal)
            )
        second_residual = torch.linalg.norm(second_moment - second_reconstruction) / max(
            float(torch.linalg.norm(second_moment).item()), 1e-12
        )
        third_residual = torch.linalg.norm(third_moment - third_reconstruction) / max(
            float(torch.linalg.norm(third_moment).item()), 1e-12
        )
        pencil_residual = float(torch.maximum(second_residual, third_residual).item())
    elif estimated_rank:
        unresolved = estimated_rank

    if moment_order == "second-only":
        terms = []
        for index in torch.nonzero(keep, as_tuple=False).flatten().tolist():
            normal = eigenvectors[:, index]
            weight = eigenvalues[index]
            readout = weight * math.sqrt(2 * math.pi) * sigma
            terms.append({
                "normal": normal.detach().cpu().tolist(),
                "offset": 0.0,
                "readout": float(readout.item())
            })

    if terms:
        features = [torch.ones(sample_count, dtype=DTYPE, device=DEVICE), *x.unbind(dim=1)]
        for term in terms:
            normal = torch.tensor(term["normal"], dtype=DTYPE, device=DEVICE)
            preactivation = x @ normal + term["offset"]
            features.append(torch.relu(preactivation))
        design = torch.stack(features, dim=1)
        coefficients = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
        base_intercept = float(coefficients[0].item())
        base_weights = coefficients[1:dimension + 1].detach().cpu().tolist()
        for index, term in enumerate(terms):
            term["readout"] = float(coefficients[dimension + 1 + index].item())
    else:
        design = torch.cat([torch.ones(sample_count, 1, dtype=DTYPE, device=DEVICE), x], dim=1)
        coefficients = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
        base_intercept = float(coefficients[0].item())
        base_weights = coefficients[1:].detach().cpu().tolist()

    torch.cuda.synchronize(DEVICE)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    allocated = int(torch.cuda.max_memory_allocated(DEVICE))
    return {
        "baseIntercept": base_intercept,
        "baseWeights": base_weights,
        "terms": terms,
        "reconstructionComplete": (
            moment_order == "second-and-third"
            and estimated_rank > 0
            and unresolved == 0
            and pencil_residual is not None
            and pencil_residual < 0.25
        ),
        "diagnostics": {
            "estimatedRank": estimated_rank,
            "rankThreshold": rank_threshold,
            "absoluteMomentEigenvalues": eigenvalues.abs().sort(descending=True).values.detach().cpu().tolist(),
            "unresolvedComponents": unresolved,
            "matrixPencilResidual": pencil_residual,
            "momentOrder": moment_order,
            "computeDevice": str(DEVICE),
            "deviceName": torch.cuda.get_device_name(DEVICE),
            "cudaMemoryAllocatedBytes": allocated,
            "momentAndInversionTimeMs": elapsed_ms
        }
    }


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
        emit({"ok": True, "model": solve(request)})
    except Exception as error:
        emit({"ok": False, "error": traceback.format_exc()})
