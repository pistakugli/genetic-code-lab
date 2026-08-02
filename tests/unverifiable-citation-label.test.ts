import { buildEvidence, divergedFields, UNVERIFIABLE_STATE } from '../scripts/label-unverifiable-citations';

/**
 * 210 delegated citations point at a shared sweep report that a later run overwrote. The
 * count was visible to `verify-delegated-citations.ts` and nowhere else, so a reader opening
 * one of those reports saw a citation to a path that exists and read it as evidence. Six of
 * them publish `pass: true` against a witness that now reads `fail`.
 *
 * The label writes that into the artifact. What it must never do is make a broken citation
 * count as anything other than broken, which is why the verifier keeps counting it and only
 * adds the disclosure requirement on top.
 */
describe('unverifiable citation labelling', () => {
  const published = { pass: true, minR2: 0.9967, totalRawOracleCalls: 150433, accessInvariant: true };

  test('only the fields that actually diverged are named', () => {
    const current = { pass: true, minR2: 0.9739, totalRawOracleCalls: 150433, accessInvariant: true };
    expect(divergedFields(published, current)).toEqual(['minR2']);
  });

  test('an unchanged target produces no divergence and so no label', () => {
    expect(divergedFields(published, { ...published })).toEqual([]);
  });

  test('a field present on one side only counts as diverged', () => {
    expect(divergedFields({ pass: true }, { pass: true, minPairedAtomJointCosine: 0.24 }))
      .toEqual(['minPairedAtomJointCosine']);
  });

  test('the record carries both readings of every diverged field', () => {
    const current = { pass: false, minR2: 0.9739 };
    const evidence = buildEvidence(published, current, divergedFields(published, current));

    expect(evidence.state).toBe(UNVERIFIABLE_STATE);
    expect(evidence.publishedVsCurrent.minR2).toEqual({ published: 0.9967, current: 0.9739 });
    expect(evidence.targetPassNow).toBe(false);
  });

  test('a published pass against a target that now fails says so in words', () => {
    const current = { pass: false };
    const evidence = buildEvidence(published, current, divergedFields(published, current));

    // This is the dangerous subset: the artifact's own verdict rests on the citation.
    expect(evidence.reason).toContain('pass while the target now reads fail');
  });

  test('a divergence that does not reverse the verdict does not claim it did', () => {
    const current = { pass: true, minR2: 0.9739 };
    const evidence = buildEvidence(published, current, divergedFields(published, current));

    expect(evidence.reason).not.toContain('pass while the target now reads fail');
  });

  test('the record never suggests the citation was repaired', () => {
    const evidence = buildEvidence(published, { pass: false }, ['pass']);

    // A label is a disclosure, not a repair. Only a rerun restores provable evidence.
    expect(evidence.repair).toContain('Rerun the lane');
  });
});
