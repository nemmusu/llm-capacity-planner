// Answers of the compatibility check when the question falls outside the matrix.
import { describe, expect, it } from 'vitest';
import { checkCompat, compatData } from '../../src/planner/compat';

describe('checkCompat outside the matrix', () => {
  it('an engine the matrix does not have', () => {
    const data = { ...compatData(), engines: [] };
    expect(checkCompat({ model: 'qwen3-8-2-4t-a95b', engine: 'vllm', gpu: 'sm90' }, data)).toMatchObject({ status: 'unknown', verifiedVersion: '', reason: expect.stringContaining('not in the matrix') });
  });
  it('a precision label no checkpoint of the model has', () => {
    expect(checkCompat({ model: 'qwen3-8-2-4t-a95b', engine: 'vllm', gpu: 'sm90', format: 'gguf' })).toMatchObject({ status: 'unknown', reason: 'No gguf checkpoint of qwen3-8-2-4t-a95b in the catalog.' });
  });
});
