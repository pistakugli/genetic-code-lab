import {
  BlackBoxReluDecompilerConfig,
  BlackBoxReluModel,
  decompileBlackBoxRelu1D,
  predictBlackBoxReluBatch,
  predictBlackBoxReluModel
} from './blackbox-relu-decompiler';

export type DeepBlackBoxRelu1DModel = BlackBoxReluModel & {
  canonical: BlackBoxReluModel;
  kinkCount: number;
  regionCount: number;
  recoveryKind: 'output-equivalent-canonical-spline';
};

export function decompileDeepBlackBoxRelu1D(
  oracle: (x: number) => number,
  config: BlackBoxReluDecompilerConfig
): DeepBlackBoxRelu1DModel {
  const canonical = decompileBlackBoxRelu1D(oracle, config);
  return {
    ...canonical,
    canonical,
    kinkCount: canonical.terms.length,
    regionCount: canonical.terms.length + 1,
    recoveryKind: 'output-equivalent-canonical-spline'
  };
}

export function predictDeepBlackBoxRelu1DModel(model: DeepBlackBoxRelu1DModel, x: number): number {
  return predictBlackBoxReluModel(model.canonical, x);
}

export function predictDeepBlackBoxRelu1DBatch(model: DeepBlackBoxRelu1DModel, xs: number[]): number[] {
  return predictBlackBoxReluBatch(model.canonical, xs);
}
