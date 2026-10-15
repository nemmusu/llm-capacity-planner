// The command end to end on the built-in models, for what the planner and the matrix decide about pipeline stages,
// the sparse-attention reserve and contexts beyond a model's limits: the live path, not only the planner's functions.
import { afterAll, describe, expect, it } from 'vitest';
import { runCli, tempFolder } from '../helpers/catalog';

const cache = tempFolder('live-path-');
afterAll(() => cache.cleanup());

interface Json {
  fit?: { figures: { pp: number } };
  rows: { pp: number; verdict: string }[];
}

describe('the command on the built-in models', { timeout: 120_000 }, () => {
  it('reads the matrix for the exact checkpoint: Gemma-4-E4B is never split into vLLM pipeline stages (wrong output)', async () => {
    const r = await runCli(['google/gemma-4-E4B-it', '--gpu', 'rtx-4090', '--json', 'g.json'], cache.dir);
    expect(r.code).toBe(0);
    const json = JSON.parse(r.written['g.json'] as string) as Json;
    expect(json.rows.filter((x) => x.pp > 1).every((x) => x.verdict === 'no-start')).toBe(true);
    expect(json.fit?.figures.pp ?? 1).toBe(1);
  });

  it('names the sparse-attention reserve when it is what puts a layout at the limit', async () => {
    const r = await runCli(['zai-org/GLM-5.3', '--tokens', '268k', '--gpu', 'h200-sxm'], cache.dir);
    expect(r.code).toBe(0);
    expect(r.text).toContain('the 21 GiB per GPU that sparse attention needs beyond 128K tokens');
    expect(r.text).not.toMatch(/Caution: it is at the limit of memory: it fits only if the engine's fixed costs/);
  });

  it('shows why a pipeline layout is refused, with the documented workaround, in every layout listed', async () => {
    const r = await runCli(['zai-org/GLM-5.3', '--tokens', '128k', '--gpu', 'h200-nvl', '--all-hw'], cache.dir);
    expect(r.code).toBe(0);
    expect(r.text).toMatch(/split in 2 stages of 4\) +engine does not start \(\d+\)/);
    expect(r.text).toContain('Workaround: VLLM_PP_LAYER_PARTITION=38,40');
  });

  it('one closest layout everywhere: the summary names the one the headline does (GLM-5.3 at 1M for 200 users)', async () => {
    const r = await runCli(['zai-org/GLM-5.3', '--users', '200', '--no-color'], cache.dir);
    expect(r.code).toBe(0);
    const closest = /Closest: 1 server with 8. NVIDIA ([^,]+),/.exec(r.text.replace(/\s+/g, ' '))?.[1];
    const named = /the closest is 8. (\S+ \S+) \(/.exec(r.text.replace(/\s+/g, ' '))?.[1];
    expect(closest).toBeDefined();
    expect(closest).toBe(named);
  });

  it("each workaround stays on its own rule: Gemma-4-E4B's pipeline bug is not followed by the fp8 KV cache's workaround", async () => {
    const r = await runCli(['google/gemma-4-E4B-it', '--gpu', 'rtx-4090', '--no-color'], cache.dir);
    expect(r.code).toBe(0);
    const pp = r.text.replace(/\s+/g, ' ').match(/Pipeline parallelism does not work.*?\(source: \S+\)/)?.[0];
    expect(pp).toBeDefined();
    expect(pp).not.toContain('TRITON_ATTN');
  });

  it('no recommendation on figures marked "do not plan", and pipeline stages said in the summary', async () => {
    const kimi = await runCli(['moonshotai/Kimi-K2.7-Code', '--tokens', '128k', '--users', '300', '--active', '30', '--gpu', 'h200', '--no-color'], cache.dir);
    expect(kimi.code).toBe(0);
    expect(kimi.text).toMatch(/Recommendation: none reliable — the speed is not reliable here/);
    const oss = await runCli(['openai/gpt-oss-120b', '--gpu', '3090', '--users', '50', '--no-color'], cache.dir);
    expect(oss.code).toBe(0);
    expect(oss.text.replace(/\s+/g, ' ')).toContain('pipeline speed not measured (its ranges are widened by half)');
    expect(oss.text.replace(/\s+/g, ' ')).toContain('pipeline speed not measured: no run with pipeline stages was measured');
  });

  it('the report says a context is outside the model limits exactly where the command does', async () => {
    const r = await runCli(['report', 'gpt-oss-120b', '--md', 'r.md'], cache.dir);
    expect(r.code).toBe(0);
    expect(r.text).toContain("Outside the model's limits: it is made for up to 128k tokens.");
    expect(r.written['r.md']).toContain("Outside the model's limits: it is made for up to 128k tokens.");
  });
});
