/**
 * The Hugging Face token, found the way huggingface_hub finds it, so gated repos work with the login
 * already on the PC (`huggingface-cli login` / `hf auth login`): HF_TOKEN (or the older
 * HUGGING_FACE_HUB_TOKEN), then the file HF_TOKEN_PATH, then $HF_HOME/token, then
 * ~/.cache/huggingface/token ($XDG_CACHE_HOME/huggingface/token when set). The token is never printed.
 */
import { readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export interface TokenSources {
  env: Record<string, string | undefined>;
  home?: string;
  read?: (file: string) => string;
}

/** The token, or undefined when there is no login. */
export function findHfToken(s: TokenSources): string | undefined {
  const env = s.env;
  const fromEnv = (env.HF_TOKEN ?? env.HUGGING_FACE_HUB_TOKEN)?.trim();
  if (fromEnv) return fromEnv;
  const home = s.home ?? os.homedir();
  const read = s.read ?? ((f: string) => readFileSync(f, 'utf8'));
  const hfHome = env.HF_HOME ?? path.join(env.XDG_CACHE_HOME ?? path.join(home, '.cache'), 'huggingface');
  for (const file of [env.HF_TOKEN_PATH, path.join(hfHome, 'token')]) {
    if (!file) continue;
    try {
      const token = read(file).trim();
      if (token) return token;
    } catch {
      // No file: no login there.
    }
  }
  return undefined;
}
