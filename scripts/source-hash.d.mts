/** The first 12 hex digits of the fingerprint of the command's sources (src/, data/*.json, package files); `except` leaves files out. */
export function sourceHash(root: string, except?: readonly string[]): string;
