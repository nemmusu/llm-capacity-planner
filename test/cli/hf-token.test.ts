import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { findHfToken } from '../../src/cli/hf-token';

const files = (map: Record<string, string>) => (f: string) => {
  const v = map[path.normalize(f)];
  if (v === undefined) throw new Error('ENOENT');
  return v;
};
const home = path.normalize('/home/u');

describe('findHfToken (huggingface_hub order)', () => {
  it('HF_TOKEN first, then the older HUGGING_FACE_HUB_TOKEN', () => {
    expect(findHfToken({ env: { HF_TOKEN: ' hf_a ', HUGGING_FACE_HUB_TOKEN: 'hf_b' }, home, read: files({}) })).toBe('hf_a');
    expect(findHfToken({ env: { HUGGING_FACE_HUB_TOKEN: 'hf_b' }, home, read: files({}) })).toBe('hf_b');
  });

  it('then HF_TOKEN_PATH, $HF_HOME/token, $XDG_CACHE_HOME/huggingface/token, ~/.cache/huggingface/token', () => {
    const t = path.join(home, 'custom-token');
    expect(findHfToken({ env: { HF_TOKEN_PATH: t }, home, read: files({ [path.normalize(t)]: 'hf_p\n' }) })).toBe('hf_p');
    const hfHome = path.join(home, 'hf');
    expect(findHfToken({ env: { HF_HOME: hfHome }, home, read: files({ [path.join(hfHome, 'token')]: 'hf_h' }) })).toBe('hf_h');
    const xdg = path.join(home, 'xdg');
    expect(findHfToken({ env: { XDG_CACHE_HOME: xdg }, home, read: files({ [path.join(xdg, 'huggingface', 'token')]: 'hf_x' }) })).toBe('hf_x');
    expect(findHfToken({ env: {}, home, read: files({ [path.join(home, '.cache', 'huggingface', 'token')]: 'hf_c' }) })).toBe('hf_c');
    // An empty token path falls through to the next place.
    expect(findHfToken({ env: { HF_TOKEN_PATH: t }, home, read: files({ [path.normalize(t)]: '  ', [path.join(home, '.cache', 'huggingface', 'token')]: 'hf_c' }) })).toBe('hf_c');
  });

  it('no login: undefined; the real file system and home are the defaults', () => {
    expect(findHfToken({ env: {}, home, read: files({}) })).toBeUndefined();
    expect(findHfToken({ env: { HF_TOKEN: 'hf_env' } })).toBe('hf_env');
    expect(findHfToken({ env: { HF_HOME: path.join(home, 'nowhere-for-sure-1234') } })).toBeUndefined();
  });
});
