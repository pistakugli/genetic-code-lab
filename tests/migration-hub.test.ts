import { MigrationHub } from '../src/core/migration-hub';

describe('MigrationHub', () => {
  test('records atomic migration counters across shared hub views', () => {
    const writer = new MigrationHub();
    const reader = new MigrationHub(writer.getSharedBuffer());

    writer.dropMigrant(0, new Uint32Array([1, 2, 3]), new Float64Array([0.5]), 0.75);
    writer.dropMigrant(1, new Uint32Array([4, 5]), new Float64Array([1.5]), 0.85);

    const fromOne = reader.pullMigrant(1);
    const fromZero = reader.pullMigrant(0);

    expect(fromOne).not.toBeNull();
    expect(fromOne?.sourceId).toBe(1);
    expect(fromOne?.fitness).toBeCloseTo(0.85);
    expect(Array.from(fromOne?.opcodes || [])).toEqual([4, 5]);

    expect(fromZero).not.toBeNull();
    expect(fromZero?.sourceId).toBe(0);
    expect(fromZero?.fitness).toBeCloseTo(0.75);
    expect(Array.from(fromZero?.opcodes || [])).toEqual([1, 2, 3]);

    expect(reader.getStats()).toEqual({
      drops: 2,
      pullAttempts: 2,
      pullSuccesses: 2
    });
    expect(writer.getStats()).toEqual(reader.getStats());
  });

  test('counts failed pull attempts without inventing migrants', () => {
    const hub = new MigrationHub();

    expect(hub.pullMigrant(99)).toBeNull();
    expect(hub.getStats()).toEqual({
      drops: 0,
      pullAttempts: 1,
      pullSuccesses: 0
    });
  });
});
