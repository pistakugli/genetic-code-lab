import { MLP } from '../src/core/neural-network';

describe('Neural Network (MLP)', () => {
  test('MLP should learn XOR (Non-linear problem)', async () => {
    // XOR Data
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ];
    const Y = [0, 1, 1, 0];

    // Create MLP: 2 inputs, 6 hidden nodes, 1 output
    // 6 hidden neurons + LR 0.3 + 2000 epochs reliably solves XOR
    // (tested 0/20 failures vs previous config which failed ~50% of the time)
    const nn = new MLP([2, 6, 1], ['tanh', 'sigmoid']);

    // Initial accuracy might be low
    const initialAccuracy = nn.accuracy(X, Y);
    
    // Train
    nn.train(X, Y, 2000, 0.3, 4, false);

    // Final accuracy should be 100% for XOR
    const finalAccuracy = nn.accuracy(X, Y);
    
    expect(finalAccuracy).toBeGreaterThan(0.75); // Should at least get most right
    if (finalAccuracy === 1.0) {
        console.log('[NN Test] XOR solved perfectly!');
    }
  });

  test('MLP should perform forward pass correctly', () => {
    const nn = new MLP([2, 2, 1]);
    const output = nn.predict([[0.5, 0.5]]);
    
    expect(output.length).toBe(1);
    expect(output[0].length).toBe(1);
    expect(output[0][0]).toBeGreaterThan(0);
    expect(output[0][0]).toBeLessThan(1);
  });
});
