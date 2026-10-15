// A model cut to its first layers, the way the measurement scripts start it in vLLM (--hf-overrides) and the
// way the planner then computes it (the same config, the tensors of the kept layers). Development only.
import type { RemoteModel } from '../../src/cli/model';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import { plannerModel } from '../../src/planner/model';

/** config.json overrides that keep the first `n` layers (and the per-layer lists that go with them). */
export function cutTo(config: Record<string, unknown>, n: number): Record<string, unknown> {
  const cut = (c: Record<string, unknown>) => {
    const o: Record<string, unknown> = { num_hidden_layers: n };
    for (const k of ['layer_types', 'swiglu_limits', 'swiglu_limits_shared', 'partial_rotary_factors', 'rope_theta', 'compress_ratios'])
      if (Array.isArray(c[k]) && (c[k] as unknown[]).length >= n) o[k] = (c[k] as unknown[]).slice(0, n);
    return o;
  };
  const text = config.text_config as Record<string, unknown> | undefined;
  return text ? { text_config: cut(text) } : cut(config);
}

/** The model as vLLM loads it with `layers` kept (undefined = whole): overrides, config, tensors, planner model. */
export function cutModel(remote: RemoteModel, layers: number | undefined) {
  const config = remote.config as Record<string, unknown>;
  const overrides = layers !== undefined ? cutTo(config, layers) : undefined;
  const cutConfig = overrides ? (JSON.parse(JSON.stringify(config)) as Record<string, unknown>) : config;
  if (overrides) {
    const target = (cutConfig.text_config ?? cutConfig) as Record<string, unknown>;
    Object.assign(target, (overrides as Record<string, unknown>).text_config ?? overrides);
  }
  const shape = loadShapeFromHfConfig(cutConfig);
  const allLayers = loadShapeFromHfConfig(config).layers ?? 0;
  // Kept: tensors outside the layers, the vision tower, the kept layers, and MTP layers past the last one.
  const keep = (name: string) => {
    const m = /(?:^|\.)layers\.(\d+)\./.exec(name);
    return !m || /vision|visual/.test(name) || layers === undefined || Number(m[1]) < layers || Number(m[1]) >= allLayers;
  };
  const tensors = remote.tensors.filter((t) => keep(t.name));
  return { overrides, config: cutConfig, shape, tensors, planner: plannerModel(cutConfig, factsFromHfConfig(cutConfig), shape, tensors) };
}
