// Servers needed for a number of users: independent copies of the model, one formula for code and docs
// (docs/METHODOLOGY.md §7.3): users per server = floor(guaranteed concurrent requests per server ÷ active share),
// servers = ceil(users ÷ users per server).
import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import type { Capacity } from '../../src/planner/capacity';
import type { Gpu } from '../../src/planner/hardware';
import { serverPlan, serversOf } from '../../src/planner/plan';

const gpu = (maxPerServer?: number) => ({ id: 'x', ...(maxPerServer ? { maxPerServer } : {}) }) as Gpu;
const row = (reliable: number, probable: number, count: number, maxPerServer?: number, verdict: 'fits' | 'borderline' = 'fits') => ({
  verdict,
  capacity: { reliable, probable } as Capacity,
  layout: { gpu: gpu(maxPerServer), count, tp: 1, pp: 1 },
});

describe('serverPlan', () => {
  it('the worked example of the review: 300 users at 30 % active, 10 requests per server → 33 per server, 10 servers', () => {
    expect(serverPlan(row(10, 18, 8), 300, 0.3)).toEqual({ servers: 10, usersPerServer: 33, likelyUsersPerServer: 60 });
  });

  it('servers × users per server always covers the users', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 5000 }), fc.integer({ min: 1, max: 400 }), fc.integer({ min: 1, max: 99 }), fc.constantFrom(1, 2, 4, 8, 16), (users, reliable, activePct, count) => {
        const p = serverPlan(row(reliable, reliable * 2, count), users, activePct / 100);
        if (!p) return;
        expect(p.servers * p.usersPerServer).toBeGreaterThanOrEqual(users);
        // And one server fewer would not.
        expect((p.servers - 1) * p.usersPerServer).toBeLessThan(users);
      }),
    );
  });

  it('a layout over two servers counts per server; nothing for a layout that does not fit or serves nobody', () => {
    // 16 GPUs of an 8-GPU server: 2 servers, 40 requests → 20 per server.
    expect(serversOf({ gpu: gpu(), count: 16 })).toBe(2);
    expect(serverPlan(row(40, 40, 16), 100, 0.1)).toMatchObject({ usersPerServer: 200, servers: 1 });
    // A card sold two per server: 4 GPUs are 2 servers.
    expect(serversOf({ gpu: gpu(2), count: 4 })).toBe(2);
    expect(serverPlan(row(0, 1, 1), 10, 0.1)).toBeUndefined();
    expect(serverPlan(row(5, 5, 1, undefined, 'borderline'), 10, 0.1)).toBeUndefined();
    expect(serverPlan({ verdict: 'fits', capacity: undefined, layout: { gpu: gpu(), count: 1, tp: 1, pp: 1 } }, 10, 0.1)).toBeUndefined();
  });
});
