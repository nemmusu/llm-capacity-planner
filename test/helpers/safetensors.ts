/** Tiny synthetic safetensors files and temporary folders for the tests (no big binaries in git). */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const DTYPE_BYTES: Record<string, number> = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1 };

export interface SafetensorsSpec {
  tensors?: Record<string, { dtype: string; shape: number[] }>;
  metadata?: Record<string, string>;
}

/** A safetensors file: u64 LE header length, the JSON header (padded to 8 bytes), zeroed tensor data. */
export function buildSafetensors({ tensors = {}, metadata }: SafetensorsSpec = {}): Buffer {
  const header: Record<string, unknown> = {};
  if (metadata) header.__metadata__ = metadata;
  let offset = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1) * (DTYPE_BYTES[t.dtype] ?? 1);
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + n] };
    offset += n;
  }
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - (json.length % 8)) % 8);
  const out = Buffer.alloc(8 + Buffer.byteLength(json) + offset);
  out.writeBigUInt64LE(BigInt(Buffer.byteLength(json)), 0);
  out.write(json, 8, 'utf8');
  return out;
}

export function writeSafetensors(file: string, spec: SafetensorsSpec = {}): string {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, buildSafetensors(spec));
  return file;
}

export function tempDir(prefix = 'planner-'): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(path.join(os.tmpdir(), prefix));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}
