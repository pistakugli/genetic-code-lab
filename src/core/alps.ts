/**
 * Genetic Code Lab - Age-Layered Population Structure (ALPS)
 * Prevents premature convergence by segregating individuals by evolutionary age.
 */

import { Genome, ALPSConfig, Population } from './types';
import { deepClone, randomInt } from '../utils/helpers';

export interface ALPSLayer {
  id: number;
  genomes: Genome[];
  maxAge: number;
}

/**
 * ALPS Manager handles layer segregation and restricted competition
 */
export class ALPSManager {
  private layers: ALPSLayer[] = [];
  private config: ALPSConfig;

  constructor(config: ALPSConfig) {
    this.config = config;
    this.initializeLayers();
  }

  /**
   * Initialize age-based layers
   */
  private initializeLayers() {
    for (let i = 0; i < this.config.numLayers; i++) {
      // Fiber layer: Max age increases exponentially or linearly
      // Traditional ALPS uses: Age(layer) = ageGap * (layer + 1)
      const maxAge = this.config.ageGap * (i + 1);
      this.layers.push({
        id: i,
        genomes: [],
        maxAge: i === this.config.numLayers - 1 ? Infinity : maxAge
      });
    }
  }

  /**
   * Segregate genomes into layers based on their age
   */
  public segregate(genomes: Genome[]) {
    // Clear layers
    this.layers.forEach(l => l.genomes = []);

    for (const genome of genomes) {
      const layerIdx = this.getLayerIndex(genome.age);
      this.layers[layerIdx].genomes.push(genome);
    }
  }

  /**
   * Determine layer index for a given age
   */
  private getLayerIndex(age: number): number {
    for (let i = 0; i < this.layers.length; i++) {
      if (age < this.layers[i].maxAge) {
        return i;
      }
    }
    return this.layers.length - 1;
  }

  /**
   * Get genomes available for selection for a specific layer.
   * ALPS allows selection from the same layer or the layer immediately below.
   */
  public getSelectionPool(layerIdx: number): Genome[] {
    if (layerIdx === 0) {
      return this.layers[0].genomes;
    }
    
    // Combine current layer and previous layer
    return [
      ...this.layers[layerIdx].genomes,
      ...this.layers[layerIdx - 1].genomes
    ];
  }

  /**
   * Get all layers
   */
  public getLayers(): ALPSLayer[] {
    return this.layers;
  }

  /**
   * Update age of all genomes and check for layer transitions
   */
  public updateAges(genomes: Genome[]) {
    for (const genome of genomes) {
      genome.age++;
    }
  }

  /**
   * Check if Layer 0 should be refreshed (Innovation Sweep)
   */
  public shouldRefreshLayer0(generation: number): boolean {
    return generation > 0 && generation % this.config.innovationInterval === 0;
  }
}

/**
 * Helper to build a layer-aware population for selection operators
 */
export function createLayerPopulation(genomes: Genome[], basePopulation: Population): Population {
  return {
    ...basePopulation,
    genomes: genomes
  };
}
