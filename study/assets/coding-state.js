(function (root) {
  'use strict';
  const valid = value => value && typeof value === 'object' && !Array.isArray(value);
  function parse(raw) {
    try { const value = JSON.parse(raw); return valid(value) ? value : {}; }
    catch (_) { return {}; }
  }
  function load(storage, course, suffix, ids) {
    const key = 'sideinn:coding:' + course + ':v1' + suffix;
    const records = {};
    let writable = true;
    try {
      const current = parse(storage.getItem(key));
      // Recognize the previous course-name format without displaying its label.
      const candidates = [];
      for (let i = 0; i < storage.length; i++) {
        const candidate = storage.key(i);
        if (candidate.startsWith('sideinn:coding:') && candidate.endsWith('-' + course + ':v1' + suffix)) candidates.push(candidate);
      }
      const previous = candidates.length === 1 ? parse(storage.getItem(candidates[0])) : {};
      let migrated = false;
      for (const id of ids) {
        if (valid(current[id])) { records[id] = current[id]; continue; }
        const number = id.split('-').pop();
        const oldId = Object.keys(previous).find(old => /^[A-Z]+-\d+$/.test(old) && old.split('-').pop() === number);
        if (oldId && valid(previous[oldId])) { records[id] = previous[oldId]; migrated = true; }
      }
      // Never remove the old copy; a failed write still leaves it recoverable.
      if (migrated) storage.setItem(key, JSON.stringify(records));
    } catch (_) { writable = false; }
    return {key, records, writable};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {load};
  else root.CodingState = {load};
})(typeof globalThis !== 'undefined' ? globalThis : this);
