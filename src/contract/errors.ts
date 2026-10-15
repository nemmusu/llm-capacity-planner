/** Failures the command reports, by code (the message is the technical detail). */
export const ERROR_CODES = ['VALIDATION', 'NETWORK', 'MODEL_INVALID', 'HF_AUTH_REQUIRED', 'HF_GATED', 'HF_RATE_LIMITED', 'HF_NOT_FOUND'] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

export class AppError extends Error {
  readonly code: ErrorCode;
  readonly params: Record<string, string | number | boolean> | undefined;
  readonly detail: string | undefined;

  constructor(code: ErrorCode, message: string, opts: { params?: Record<string, string | number | boolean>; detail?: string; cause?: unknown } = {}) {
    super(message, opts.cause === undefined ? undefined : { cause: opts.cause });
    this.name = 'AppError';
    this.code = code;
    this.params = opts.params;
    this.detail = opts.detail;
  }
}

/** The message of anything thrown. */
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
