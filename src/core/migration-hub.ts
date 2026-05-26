import { OpCode } from './pop-buffer';

/**
 * PHASE 41: SHARED MEMORY ADAPTIVE MIGRATION HUB
 * Enables zero-copy, source-aware genotype exchange between worker islands.
 */
export class MigrationHub {
    private sharedBuffer: SharedArrayBuffer;
    private view: Int32Array;
    private floatView: Float64Array;
    
    private readonly MAX_MIGRANTS = 512;
    private readonly GENOME_SIZE = 256; 
    private readonly CONST_SIZE = 10;
    private readonly SLOT_BYTES = (256 * 4) + (10 * 8) + 16; 
    private readonly ATOMIC_DROPS = 0;
    private readonly ATOMIC_PULL_ATTEMPTS = 1;
    private readonly ATOMIC_PULL_SUCCESSES = 2;

    constructor(buffer?: SharedArrayBuffer) {
        if (buffer) {
            this.sharedBuffer = buffer;
        } else {
            this.sharedBuffer = new SharedArrayBuffer(1024 + (this.MAX_MIGRANTS * this.SLOT_BYTES));
        }
        this.view = new Int32Array(this.sharedBuffer);
        this.floatView = new Float64Array(this.sharedBuffer);
    }

    public getSharedBuffer(): SharedArrayBuffer {
        return this.sharedBuffer;
    }

    /**
     * Drop a champion into a random slot in the hub.
     * Updated for Phase 41: Source Island ID recorded.
     */
    public dropMigrant(islandId: number, opcodes: Uint32Array, constants: Float64Array, fitness: number, generation: number = 0): void {
        const dropIndex = Atomics.add(this.view, this.ATOMIC_DROPS, 1);
        const slot = this.slotFor(dropIndex, islandId);
        const offset = 256 + (slot * (this.SLOT_BYTES / 4)); 

        // 1. Write Metadata (Fitness, Len, SourceID, SourceGeneration)
        this.floatView[offset / 2] = fitness; 
        this.view[offset + 2] = opcodes.length;
        this.view[offset + 3] = islandId; // Phase 41: Tracking source
        this.view[offset + 4] = generation; // birth generation of this migrant

        // 2. Write Instructions
        for (let i = 0; i < opcodes.length && i < 256; i++) {
            this.view[offset + 5 + i] = opcodes[i];
        }

        // 3. Write Constants
        const constOffset = offset + 261; 
        for (let i = 0; i < constants.length && i < 10; i++) {
            this.floatView[(constOffset / 2) + i] = constants[i];
        }
    }

    /**
     * Pull a migrant from the hub.
     * Updated for Phase 41: Supports source-based preference.
     */
    public pullMigrant(sourcePreference?: number): { opcodes: Uint32Array, constants: Float64Array, fitness: number, sourceId: number, sourceGeneration: number } | null {
        const pullIndex = Atomics.add(this.view, this.ATOMIC_PULL_ATTEMPTS, 1);
        // Linear scan for a matching slot or deterministic fall-through.
        let slot = this.slotFor(pullIndex, sourcePreference ?? 0);
        
        if (sourcePreference !== undefined) {
            for (let i = 0; i < this.MAX_MIGRANTS; i++) {
                const sOffset = 256 + (i * (this.SLOT_BYTES / 4));
                if (this.view[sOffset + 3] === sourcePreference) {
                    slot = i;
                    break;
                }
            }
        }

        const offset = 256 + (slot * (this.SLOT_BYTES / 4));
        const fitness = this.floatView[offset / 2];
        if (fitness === 0 || isNaN(fitness) || fitness === -Infinity) return null;

        const len = this.view[offset + 2];
        const sourceId = this.view[offset + 3];
        const sourceGeneration = this.view[offset + 4];
        if (len <= 0 || len > 256) return null;

        const opcodes = new Uint32Array(len);
        for (let i = 0; i < len; i++) opcodes[i] = this.view[offset + 5 + i];

        const constants = new Float64Array(10);
        const constOffset = offset + 261;
        for (let i = 0; i < 10; i++) constants[i] = this.floatView[(constOffset / 2) + i];

        Atomics.add(this.view, this.ATOMIC_PULL_SUCCESSES, 1);
        return { opcodes, constants, fitness, sourceId, sourceGeneration };
    }

    private slotFor(counter: number, salt: number): number {
        return Math.abs((counter + salt * 131) % this.MAX_MIGRANTS);
    }

    public getStats(): { drops: number; pullAttempts: number; pullSuccesses: number } {
        return {
            drops: Atomics.load(this.view, this.ATOMIC_DROPS),
            pullAttempts: Atomics.load(this.view, this.ATOMIC_PULL_ATTEMPTS),
            pullSuccesses: Atomics.load(this.view, this.ATOMIC_PULL_SUCCESSES)
        };
    }
}
