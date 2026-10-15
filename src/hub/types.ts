/** One file of a Hugging Face repo. */
export interface HubFile {
  path: string;
  sizeBytes: number;
  /** LFS sha256, when the hub gives it. */
  sha256?: string;
  kind: 'weights' | 'mtp' | 'draft' | 'config' | 'tokenizer' | 'other';
}

/** A model repo as the hub's API describes it. */
export interface HubModel {
  repo: string;
  author: string;
  name: string;
  gated: boolean | string;
  license?: string;
  /** Formats the repo ships: safetensors (vLLM / SGLang) or something else. */
  formats: ('safetensors' | 'other')[];
  paramsB?: number;
  pipelineTag?: string;
  tags: string[];
}
