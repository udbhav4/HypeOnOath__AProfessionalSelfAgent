// Class with several methods + a top-level function (qualified symbols, R4 fixture).
export class Cache {
  private store = new Map<string, number>();

  get(key: string): number | undefined {
    return this.store.get(key);
  }

  set(key: string, value: number): void {
    this.store.set(key, value);
  }

  clear(): void {
    this.store.clear();
  }
}

export function makeCache(): Cache {
  return new Cache();
}
